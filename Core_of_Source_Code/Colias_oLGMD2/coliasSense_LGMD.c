/*
 * Description: oLGMD2
 * File: coliasSence_LGMD.c
 * Author: Qinbing Fu
 * Date: 2019 Aug
*/


#include "coliasSense_LGMD.h"
#include "coliasSense_board.h"
#include "delay.h"
#include <math.h>
#include <arm_math.h>
#include <stdlib.h>


uint16_t Image[3][Image_Height][Image_Width];   //store three frames of the initial image
int8_t Diff_Image[2][Image_Height][Image_Width];
LGMD_pControlTypedef hLGMD;

uint8_t LGMD_Param_Init(LGMD_pControlTypedef* hLGMD)
{
	float on_base_tau = 15;
	float off_base_tau = 60;
	LGMDstruct_Params *Params = &hLGMD->Model->Params;
	Params->Tpm = 8;////////////////////////////////////////////////////////////////////////////////////////////////
	Params->On_Tpm = 0.5;
	Params->Off_Tpm = 1;
	Params->Cw = 4;
	Params->CdeTde = 35;
	Params->n = 0;
	Params->flag = 0;
	Params->a4 = 4;
	Params->Clip = 0;
	Params->Delatc = 0.01;
	Params->a1 = 0.1;
	Params->ffi_tau = 10;
	Params->Won_base = 1;
	Params->Woff_base = 0.3;
	Params->o1 = 0.5;
	Params->o2 = 1;
	Params->o3 = 1;
	Params->Kspi = 0.5;      //0.5~1
	Params->Tspi = 0.63;      //0.65~0.78///////////////////////////////////////////////////////////////////////////////////
	Params->Tsfa = 0.003;
	Params->sfa_tau = 500;   //500~1000
	
	hLGMD->currentDiffImage=0;
	hLGMD->currentImage=1;
	hLGMD->Enable=1;
	hLGMD->Model=&LGMD;
	hLGMD->processCount=1;
	hLGMD->processCountLast=1;
	hLGMD->status=0;
	hLGMD->processRate=30;
	hLGMD->AGC_enable_period=0;
	
	Params->interval = 1000/hLGMD->processRate;
	Params->a2 = Params->interval/(Params->interval+Params->ffi_tau);  //delay coefficient in PM pathway
	Params->a3 = Params->sfa_tau/(Params->sfa_tau+Params->interval);   //delay coefficient in SFA mechanism
	Params->on_center_tau = on_base_tau;
	Params->on_near_tau = 2*on_base_tau;
	Params->on_diag_tau = 3*on_base_tau;
	Params->on_delay_center = Params->interval/(Params->interval+Params->on_center_tau);
	Params->on_delay_near = Params->interval/(Params->interval+Params->on_near_tau);
	Params->on_delay_diag = Params->interval/(Params->interval+Params->on_diag_tau);
	Params->off_center_tau = off_base_tau;
	Params->off_near_tau = 2*off_base_tau;
	Params->off_diag_tau = 3*off_base_tau;
	Params->off_delay_center = Params->interval/(Params->interval+Params->off_center_tau);
	Params->off_delay_near = Params->interval/(Params->interval+Params->off_near_tau);
	Params->off_delay_diag = Params->interval/(Params->interval+Params->off_diag_tau);
	
	printf(" LGMD's parameter initialization is complete");
	return 0;
}

uint8_t Calc_LGMDs_OutOfChannel(LGMD_pControlTypedef* hLGMD)
{
	uint8_t Im[3] = {2,0,1};
	uint8_t Im1[3] = {1,2,0};
	int8_t *diff_image;
	uint16_t ncell,n;
	uint16_t *cur_image,*pre_image;
	int16_t temg;
	
  hLGMD->currentDiffImage = !hLGMD->currentDiffImage;
	ncell = Image_Width*Image_Height;
	cur_image = (uint16_t*)(((uint8_t*)&Image[Im[(*(hLGMD->hFrameCount))%3]][0][0])+1);
  pre_image = (uint16_t*)(((uint8_t*)&Image[Im1[(*(hLGMD->hFrameCount))%3]][0][0])+1);
	diff_image = &Diff_Image[hLGMD->currentDiffImage][0][0];
	
	for(n=0;n<ncell;n++)
	{
		temg = (*cur_image - *pre_image)>>9;
		if(temg>127)
			temg = 127;
		if(temg<-127)
			temg = -127;
		*diff_image = temg;
		cur_image++;
		pre_image++;
		diff_image++;
	}
	return 0;
}

uint8_t Fast_LGMD(LGMD_pControlTypedef* hLGMD)
{
	LGMDstruct_Layer *Layers = &hLGMD->Model->Layers;
	LGMDstruct_Params *Params = &hLGMD->Model->Params;
	LGMDstruct_Result *Results = &hLGMD->Model->Results;
	int8_t *diff_image;
	//uint8_t On_Tpm,Off_Tpm,Tpm;
	uint8_t Clip;
	uint8_t Tpm;
	uint8_t cur_image;
	uint8_t width,height,i,j;
	uint8_t *cur_Pon,*pre_Pon,*cur_Poff,*pre_Poff;
	uint8_t *layS,*layG,*lay_Son,*lay_Soff;
	uint8_t t_max,Cw,CdeTde,a4;
	uint8_t *flag,*num,*spike;
	uint16_t ncell,n,center,near,diag,inhibition1,inhibition2;
	uint16_t pre_center,pre_near,pre_diag,sum;
	float a1,*w1,*w2,Son,Soff,o1,o2,o3,Delatc,scale,a3;
	float on_center,on_near,on_diag,off_center,off_near,off_diag;
	float *mp,*cur_smp,*pre_smp,*cur_lgmd_out,*pre_lgmd_out;
	float Kspi,Tsfa,Tspi;
	float on_sum_ffi,off_sum_ffi,ffi,ffi1,a2;
	float *cur_ffi,*pre_ffi,*cur_ffi1,*pre_ffi1;
	float *Won_base,*Woff_base;
	float On_Tpm,Off_Tpm;
	
	on_sum_ffi = 0;
	off_sum_ffi = 0;
	Won_base = &hLGMD->Model->Params.Won_base;
	Woff_base = &hLGMD->Model->Params.Woff_base;
	Clip = Params->Clip;
	On_Tpm = Params->On_Tpm;
	Off_Tpm = Params->Off_Tpm;
	Tpm = Params->Tpm;
	a2 = Params->a2;
	height = Image_Height;
	width = Image_Width;
	ncell = height*width;
	t_max = 0;
	a1 = Params->a1;
	w1 = &Params->w1;
	w2 = &Params->w2;
	o1 = Params->o1;
	o2 = Params->o2;
	o3 = Params->o3;
	Cw = Params->Cw;
	a3 = Params->a3;
	a4 = Params->a4;
	CdeTde = Params->CdeTde;
	Delatc = Params->Delatc;
	Kspi = Params->Kspi;
	Tsfa = Params->Tsfa;
	Tspi = Params->Tspi;
	mp = &Results->MP;
	flag = &Params->flag;
	num = &Params->n;
	spike = &Results->SPIKE;
	cur_image = hLGMD->currentDiffImage;
	on_center = Params->on_delay_center;
	on_near = Params->on_delay_near;
	on_diag = Params->on_delay_diag;
	off_center = Params->off_delay_center;
	off_near = Params->off_delay_near;
	off_diag = Params->off_delay_diag;
	cur_ffi = &Results->FFI_out[cur_image];     //in ON chaanel
	pre_ffi = &Results->FFI_out[!cur_image];    //in ON channel
	cur_ffi1 = &Results->FFI_out1[cur_image];   //in OFF channel
	pre_ffi1 = &Results->FFI_out1[!cur_image];  //in OFF channel
	diff_image = &Diff_Image[cur_image][0][0];
	cur_Pon = &Layers->ON[cur_image][0][0];
	pre_Pon = &Layers->ON[!cur_image][0][0];
	cur_Poff = &Layers->OFF[cur_image][0][0];
	pre_Poff = &Layers->OFF[!cur_image][0][0];
	cur_smp = &Results->SMP[cur_image];
	pre_smp = &Results->SMP[!cur_image];
	cur_lgmd_out = &Results->LGMD_out[cur_image];
	pre_lgmd_out = &Results->LGMD_out[!cur_image];
	layS = &Layers->S[0][0];
	layG = &Layers->G[0][0];
	lay_Son = &Layers->S_ON[0][0];
	lay_Soff = &Layers->S_OFF[0][0];
	*mp = 0;
	*flag = 0;
	
	//half-wave rectifying mechanism
  	for(n=0;n<ncell;n++)
	{
		if(*diff_image>=0)
		{
			//*cur_Pon = (uint8_t)(*diff_image+a1*(*pre_Pon));
			//*cur_Poff = (uint8_t)(0+a1*(*pre_Poff));
			*cur_Pon = *diff_image;
			*cur_Poff = 0;
		}
		else
		{
			//*cur_Pon = (uint8_t)(0+a1*(*pre_Pon));
			//*cur_Poff = (uint8_t)(abs(*diff_image)+a1*(*pre_Poff));
			*cur_Pon = 0;
			*cur_Poff = abs(*diff_image);
		}
		on_sum_ffi += (*cur_Pon);
		off_sum_ffi += (*cur_Poff);
		diff_image++;
		cur_Pon++;
		pre_Pon++;
		cur_Poff++;
		pre_Poff++; 
	}
	
	*cur_ffi = on_sum_ffi/ncell;
	ffi = a2*(*cur_ffi)+(1-a2)*(*pre_ffi);     //in ON channel
	*cur_ffi1 = off_sum_ffi/ncell;
	ffi1 = a2*(*cur_ffi1)+(1-a2)*(*pre_ffi1);  //in OFF channel
	*w1 = (ffi)/Tpm;
	if(*Won_base>*w1)
		*w1 = *Won_base;
	*w2 = (ffi1)/Tpm;
	if(*Woff_base>*w2)
		*w2 = *Woff_base; 
	
	if(ffi<On_Tpm)
	{
		for(i=1;i<height-1;i++)
		{
			lay_Son = &Layers->S_ON[i][0];
			for(j=1;j<width-1;j++)
			{
				*(lay_Son+j) = 0;
			}
		}
	}
	else
	{
		for(i=1;i<height-1;i++)
		{
			cur_Pon = &Layers->ON[cur_image][i][0];
		  pre_Pon = &Layers->ON[!cur_image][i][0];
			lay_Son = &Layers->S_ON[i][0];
			for(j=1;j<width-1;j++)
			{
				//calculate the inhibition of ON channel
			  center = (*(cur_Pon+j))<<1;
			  near = (*(cur_Pon+j-width)>>1)+(*(cur_Pon+j-1)>>1)+(*(cur_Pon+j+1)>>1)+(*(cur_Pon+j+width)>>1);
			  diag = (*(cur_Pon+j-width-1)>>2)+(*(cur_Pon+j-width+1)>>2)+(*(cur_Pon+j+width-1)>>2)+(*(cur_Pon+j+width+1)>>2);
			  pre_center = (*(pre_Pon+j))<<1;
			  pre_near = (*(pre_Pon+j-width)>>1)+(*(pre_Pon+j-1)>>1)+(*(pre_Pon+j+1)>>1)+(*(pre_Pon+j+width)>>1);
			  pre_diag = (*(pre_Pon+j-width-1)>>2)+(*(pre_Pon+j-width+1)>>2)+(*(pre_Pon+j+width-1)>>2)+(*(pre_Pon+j+width+1)>>2);
			  inhibition1 = (uint16_t)(center*on_center+near*on_near+diag*on_diag+pre_center*(1-on_center)+pre_near*(1-on_near)+pre_diag*(1-on_diag));
			
			  //calculate lay_Son
			  *(lay_Son+j) = (uint8_t)(*(cur_Pon+j)-(*w1)*inhibition1);
			  if(*(lay_Son+j)<Clip)
					*(lay_Son+j) = 0;
			}
		}
	}
	if(ffi1<Off_Tpm)
	{
		for(i=1;i<height-1;i++)
		{
			lay_Soff = &Layers->S_OFF[i][0];
			for(j=1;j<width-1;j++)
			{
				*(lay_Soff+j) = 0;
			}
		}
	}
	else
	{
		for(i=1;i<height-1;i++)
		{
		  cur_Poff = &Layers->OFF[cur_image][i][0];
		  pre_Poff = &Layers->OFF[!cur_image][i][0];
		  lay_Soff = &Layers->S_OFF[i][0];
			for(j=1;j<width-1;j++)
			{
			  //calculate the inhibition of OFF chanel
			  center = *(cur_Poff+j);
			  near = (*(cur_Poff+j-width)>>2)+(*(cur_Poff+j-1)>>2)+(*(cur_Poff+j+1)>>2)+(*(cur_Poff+j+width)>>2);
			  diag = (*(cur_Poff+j-width-1)>>3)+(*(cur_Poff+j-width+1)>>3)+(*(cur_Poff+j+width-1)>>3)+(*(cur_Poff+j+width+1)>>3);
        	  pre_center = *(pre_Poff+j);
			  pre_near = (*(pre_Poff+j-width)>>2)+(*(pre_Poff+j-1)>>2)+(*(pre_Poff+j+1)>>2)+(*(pre_Poff+j+width)>>2);
			  pre_diag = (*(pre_Poff+j-width-1)>>3)+(*(pre_Poff+j-width+1)>>3)+(*(pre_Poff+j+width-1)>>3)+(*(pre_Poff+j+width+1)>>3);
			  inhibition2 = (uint16_t)(center*off_center+near*off_near+diag*off_diag+pre_center*(1-off_center)+pre_near*(1-off_near)+pre_diag*(1-off_diag));
			
			  //calculate lay_Soff
			  *(lay_Soff+j) = (uint8_t)(*(cur_Poff+j)-(*w2)*inhibition2);
			  if(*(lay_Soff+j)<Clip)
					*(lay_Soff+j) = 0;
			}
		}
	}
	
	//Calculate layS
	for(i=1;i<height-1;i++)
	{
		lay_Son = &Layers->S_ON[i][0];
		lay_Soff = &Layers->S_OFF[i][0];
		layS = &Layers->S[i][0];
		for(j=1;j<width-1;j++)
		{
			*(layS+j) = (uint8_t)(o1*(*(lay_Son+j))+o2*(*(lay_Soff+j)));
		}	
	}
	
	/*
	if(ffi>On_Tpm && ffi1>Off_Tpm)
	{
		for(i=1;i<height-1;i++)
		{
			cur_Pon = &Layers->ON[cur_image][i][0];
		  pre_Pon = &Layers->ON[!cur_image][i][0];
		  cur_Poff = &Layers->OFF[cur_image][i][0];
		  pre_Poff = &Layers->OFF[!cur_image][i][0];
		  layS = &Layers->S[i][0];
			for(j=1;j<width-1;j++)
			{
				//calculate the inhibition of ON channel
			  center = (*(cur_Pon+j))<<1;
			  near = (*(cur_Pon+j-width)>>1)+(*(cur_Pon+j-1)>>1)+(*(cur_Pon+j+1)>>1)+(*(cur_Pon+j+width)>>1);
			  diag = (*(cur_Pon+j-width-1)>>2)+(*(cur_Pon+j-width+1)>>2)+(*(cur_Pon+j+width-1)>>2)+(*(cur_Pon+j+width+1)>>2);
			  pre_center = (*(pre_Pon+j))<<1;
			  pre_near = (*(pre_Pon+j-width)>>1)+(*(pre_Pon+j-1)>>1)+(*(pre_Pon+j+1)>>1)+(*(pre_Pon+j+width)>>1);
			  pre_diag = (*(pre_Pon+j-width-1)>>2)+(*(pre_Pon+j-width+1)>>2)+(*(pre_Pon+j+width-1)>>2)+(*(pre_Pon+j+width+1)>>2);
			  inhibition1 = (uint16_t)(center*on_center+near*on_near+diag*on_diag+pre_center*(1-on_center)+pre_near*(1-on_near)+pre_diag*(1-on_diag));
			
			  //calculate the inhibition of OFF chanel
			  center = *(cur_Poff+j);
			  near = (*(cur_Poff+j-width)>>2)+(*(cur_Poff+j-1)>>2)+(*(cur_Poff+j+1)>>2)+(*(cur_Poff+j+width)>>2);
			  diag = (*(cur_Poff+j-width-1)>>3)+(*(cur_Poff+j-width+1)>>3)+(*(cur_Poff+j+width-1)>>3)+(*(cur_Poff+j+width+1)>>3);
        pre_center = *(pre_Poff+j);
			  pre_near = (*(pre_Poff+j-width)>>2)+(*(pre_Poff+j-1)>>2)+(*(pre_Poff+j+1)>>2)+(*(pre_Poff+j+width)>>2);
			  pre_diag = (*(pre_Poff+j-width-1)>>3)+(*(pre_Poff+j-width+1)>>3)+(*(pre_Poff+j+width-1)>>3)+(*(pre_Poff+j+width+1)>>3);
			  inhibition2 = (uint16_t)(center*off_center+near*off_near+diag*off_diag+pre_center*(1-off_center)+pre_near*(1-off_near)+pre_diag*(1-off_diag));
			
			  //calculate layS
			  Son = *(cur_Pon+j)-(*w1)*inhibition1;
			  if(Son<0)
					Son = 0;
			  Soff = *(cur_Poff+j)-(*w2)*inhibition2;
			  if(Soff<0)
					Soff = 0;
			  *(layS+j) = (uint8_t)(o1*Son+o2*Soff);
			}
		}
	}
	else
	{
		if(ffi<On_Tpm)
		{
			for(i=1;i<height-1;i++)
			{
				cur_Poff = &Layers->OFF[cur_image][i][0];
		    pre_Poff = &Layers->OFF[!cur_image][i][0];
		    layS = &Layers->S[i][0];
				for(j=1;j<width-1;j++)
				{
					//calculate the inhibition of OFF chanel
			    center = *(cur_Poff+j);
			    near = (*(cur_Poff+j-width)>>2)+(*(cur_Poff+j-1)>>2)+(*(cur_Poff+j+1)>>2)+(*(cur_Poff+j+width)>>2);
			    diag = (*(cur_Poff+j-width-1)>>3)+(*(cur_Poff+j-width+1)>>3)+(*(cur_Poff+j+width-1)>>3)+(*(cur_Poff+j+width+1)>>3);
          pre_center = *(pre_Poff+j);
			    pre_near = (*(pre_Poff+j-width)>>2)+(*(pre_Poff+j-1)>>2)+(*(pre_Poff+j+1)>>2)+(*(pre_Poff+j+width)>>2);
			    pre_diag = (*(pre_Poff+j-width-1)>>3)+(*(pre_Poff+j-width+1)>>3)+(*(pre_Poff+j+width-1)>>3)+(*(pre_Poff+j+width+1)>>3);
			    inhibition2 = (uint16_t)(center*off_center+near*off_near+diag*off_diag+pre_center*(1-off_center)+pre_near*(1-off_near)+pre_diag*(1-off_diag));
					
					//calculate layS
			    Son = 0;
			    Soff = *(cur_Poff+j)-(*w2)*inhibition2;
			    if(Soff<0)
						Soff = 0;
			    *(layS+j) = (uint8_t)(o1*Son+o2*Soff);
				}
			}
		}
		else if(ffi1<Off_Tpm)
		{
			for(i=1;i<height-1;i++)
			{
				cur_Pon = &Layers->ON[cur_image][i][0];
		    pre_Pon = &Layers->ON[!cur_image][i][0];
		    layS = &Layers->S[i][0];
				for(j=1;j<width-1;j++)
				{
					//calculate the inhibition of ON channel
			    center = (*(cur_Pon+j))<<1;
			    near = (*(cur_Pon+j-width)>>1)+(*(cur_Pon+j-1)>>1)+(*(cur_Pon+j+1)>>1)+(*(cur_Pon+j+width)>>1);
			    diag = (*(cur_Pon+j-width-1)>>2)+(*(cur_Pon+j-width+1)>>2)+(*(cur_Pon+j+width-1)>>2)+(*(cur_Pon+j+width+1)>>2);
			    pre_center = (*(pre_Pon+j))<<1;
			    pre_near = (*(pre_Pon+j-width)>>1)+(*(pre_Pon+j-1)>>1)+(*(pre_Pon+j+1)>>1)+(*(pre_Pon+j+width)>>1);
			    pre_diag = (*(pre_Pon+j-width-1)>>2)+(*(pre_Pon+j-width+1)>>2)+(*(pre_Pon+j+width-1)>>2)+(*(pre_Pon+j+width+1)>>2);
			    inhibition1 = (uint16_t)(center*on_center+near*on_near+diag*on_diag+pre_center*(1-on_center)+pre_near*(1-on_near)+pre_diag*(1-on_diag));
					
					//calculate layS
			    Son = *(cur_Pon+j)-(*w1)*inhibition1;
			    if(Son<0)
					  Son = 0;
			    Soff = 0;
			    *(layS+j) = (uint8_t)(o1*Son+o2*Soff);
				}
			}
		}
		else
		{
			for(i=1;i<height-1;i++)
			{
				layS = &Layers->S[i][0];
				for(j=1;j<width-1;j++)
				{
					*(layS+j) = 0;
				}
			}
		}	
	}
  */
	
	//calculate Ce
	for(i=1;i<height-1;i++)
	{
		layG = &Layers->G[i][0];
		layS = &Layers->S[i][0];
		for(j=1;j<width-1;j++)
		{
			sum = *(layS+j)+*(layS+j-width)+*(layS+j-1)+*(layS+j+1)+*(layS+j+width)+*(layS+j-width-1)+*(layS+j-width+1)+*(layS+j+width-1)+*(layS+j+width+1);
			*(layG+j) = (uint8_t)(sum/9);
			if((*(layG+j))>t_max)
			t_max = *(layG+j);
		}
	}

	//calculate G
	scale = Delatc+(t_max*1.0)/Cw;
	for(i=1;i<height-1;i++)
	{
		layS = &Layers->S[i][0];
		layG = &Layers->G[i][0];
		for(j=1;j<width-1;j++)
		{
			*(layG+j) = (uint8_t)((*(layS+j))*(*(layG+j))*(1/scale));
			if(*(layG+j)<CdeTde)
				*(layG+j) = 0;
			*mp += *(layG+j);
		}
	}
	
	// sigmoid membrane potential
	*cur_smp = 1/(1+exp(-(*mp)/(ncell*Kspi)));
	
	//SFA mechanism
	if((*cur_smp-*pre_smp)<=Tsfa)
		*cur_lgmd_out = a3*(*(pre_lgmd_out)+*(cur_smp)-*(pre_smp));
	else
		*cur_lgmd_out = a3*(*cur_smp);
	if(*cur_lgmd_out<0.5)
		*cur_lgmd_out = 0.5;
	
	//detect collision
	*spike = floor(exp(a4*(*cur_lgmd_out-Tspi)));
	if((*spike)!=0)
	{
		*num += *spike;
		if(*num>=4)
		{
			*flag = 1;
			*num = 0;
		}
	}
	else
	{
		*num = 0;
	}
	
}

uint8_t Decision_making(LGMD_pControlTypedef* hLGMD,uint8_t allow_motion)
{
	uint8_t command = 0;
	uint8_t collision;
	collision = hLGMD->Model->Params.flag;
	uint8_t fCount=*hLGMD->hFrameCount%3;

	if (collision)	//Collision recognition
	{
		LED2_Toggle;
		if(fCount==0)
		{
			command='L';		//long right
		}
		else
		{
			command='R';		//long left
		}
		//attention
		
	}
	if(allow_motion)
	{
		/************* Pay Attention! ******************/
		//if motion sequence is not empty, do not push motions
		uint8_t readMotionQueueStatus[10]={0};
		ReadBytesWithACKFromColiasBasic(hCoS->hHCos_Motion->Instance->hHUART,0x97,readMotionQueueStatus,1);
		if((readMotionQueueStatus[1]&0x0f)==0x09)
		{
			fast_motion_control(hCoS->hHCos_Motion->Instance,command,0);	
		}
	}
	return 0;
}

uint8_t LGMD_demo(LGMD_pControlTypedef* hLGMD)
{
	uint8_t tmpOT;
	if(hLGMD->Enable==0)
		return 0;
	while (*(hLGMD ->hFrameCount) == hLGMD->processCountLast)
		hLGMD->status=0;	//wait until next frame if processing duration is less than one frame
	/*
	if (hLGMD->processRate)
		while ((*(hLGMD->hFrameCount))*(hLGMD->processRate) < (hLGMD->processCount)*OV7670_FPS)	//wait until expected processing rate	
			hLGMD->status=0;	//wait until next frame if processing duration is less than one frame
	*/
	hLGMD->status=1;
	TICin2; //start clocking,reset timer to 0
	hCoS->hHBIO->Instance->timerlog[0]=TOCin2; //save timer tick
	Calc_LGMDs_OutOfChannel(hLGMD);
	hCoS->hHBIO->Instance->timerlog[1]=TOCin2; //save timer tick
	Fast_LGMD(hLGMD);
	hCoS->hHBIO->Instance->timerlog[2]=TOCin2; //save timer tick
	//delay_ms(10);
	
	//print elapsed time
	//printf("%d \r\n",hCoS->hHBIO->Instance->timerlog[2]-hCoS->hHBIO->Instance->timerlog[0]);
	
	hLGMD->processCountLast = *(hLGMD ->hFrameCount);
	(hLGMD->processCount)++;
	//hCoS->hHBIO->Instance->timerlog[2]=TOCin2; //save timer tick
	tmpOT=256*((hLGMD->Model->Results.LGMD_out[hLGMD->currentDiffImage])-0.5);
	
	/*
	For Qinbing Attention
	the code below is to set the LEDs on the extend board,
	which is the last 8 bits of GPIOD->ODR register.
	LED is on when bit state is reset
	*/
	GPIOD->ODR=(GPIOD->ODR&0xffffff00)+(~tmpOT);
	
	return 0;
}
