/*
 * Description: oLGMD1
 * File: coliasSence_LGMD.c
 * Author: Qinbing Fu , Renyuan Liu
 * Date: 2024 May
 *   /********** / : highlights changes in this "FFI_ON_OFF" version versus normal "ON_OFF" version.
*/ 

 
#include "coliasSense_LGMD.h"
#include "coliasSense_board.h" 
#include "delay.h"
#include <math.h>
#include <arm_math.h>
#include <stdlib.h>

 
uint16_t Image[3][Image_Height][Image_Width];
int8_t Diff_Image[2][Image_Height][Image_Width];
LGMD_pControlTypedef hLGMD;

// Founctions used in LGMD_Basic listed below : 
uint8_t LGMD_Param_Init(LGMD_pControlTypedef* hLGMD)
{
	// simplize ptr passing : 
	LGMDstruct_Params* Params = &hLGMD->Model->Params;

	// LGMD general handle params :
	hLGMD->currentDiffImage = 0 ;
	hLGMD->currentImage = 1 ;
	hLGMD->Enable = 1 ;
	hLGMD->Model = &LGMD ;
	hLGMD->processCount = 1 ;
	hLGMD->processCountLast = 1 ;
	hLGMD->status = 0 ;
	hLGMD->processRate = 30 ; // 30 frames per second !
	hLGMD->AGC_enable_period = 0 ;
	
	// ON_OFF LGMD params :

	
    // tau : 
    float time_unit = 15 ; // ms
	// Params->tau_ffi_lowpass = time_unit ;
	/**********************************************************************************************/
	Params->alpha_left_on = 0.1 ;
	Params->alpha_left_off = 0.1 ;
	Params->tau_ffi_on = 10 ;
	Params->tau_ffi_off = 10 ;
	/**********************************************************************************************/
	Params->tau_sfa_highpass = 500 ;

	Params->tau_on_self_lowpass = time_unit ;
	Params->tau_on_adj_lowpass = 2*time_unit ;
	Params->tau_on_diag_lowpass = 3*time_unit ; // ms

	Params->tau_off_self_lowpass = 4*time_unit ;
	Params->tau_off_adj_lowpass = 2*4*time_unit ;
	Params->tau_on_diag_lowpass = 3*4*time_unit ; // ms

	Params->tau_in = 1000 / hLGMD->processRate ; // ms
	float tau_in = Params->tau_in ;

	// FFI_alpha : 
	// Params->alpha_ffi = tau_in/(tau_in + Params->tau_ffi_lowpass ) ;
	/**********************************************************************************************/
	Params->alpha_ffi_on = tau_in/(tau_in + Params->tau_ffi_on ) ;
	Params->alpha_ffi_off = tau_in/(tau_in + Params->tau_ffi_off ) ;

	Params->TH_on = 1 ;
	Params->TH_off = 1 ;
	/**********************************************************************************************/

	// SFA_alpha : 
	// Params->alpha_sfa = tau_in/(tau_in + Params->tau_sfa_highpass ) ; 
	Params->alpha_sfa = Params->tau_sfa_highpass/(tau_in + Params->tau_sfa_highpass ) ; 

	// ON_OFF_Convolution_alpha : 
	Params->alpha_on_self = tau_in/(tau_in + Params->tau_on_self_lowpass ) ;
	Params->alpha_on_adj  = tau_in/(tau_in + Params->tau_on_adj_lowpass ) ;
	Params->alpha_on_diag = tau_in/(tau_in + Params->tau_on_diag_lowpass ) ;

	Params->alpha_off_self = tau_in/(tau_in + Params->tau_off_self_lowpass ) ;
	Params->alpha_off_adj  = tau_in/(tau_in + Params->tau_off_adj_lowpass ) ;
	Params->alpha_off_diag = tau_in/(tau_in + Params->tau_off_diag_lowpass ) ;

	// theta: superliner cumputation : 
	Params->theta_1 = 1 ;
	Params->theta_2 = 1 ;
	Params->theta_3 = 0 ;

	Params->clip = 0 ;
    Params->Cw = 4 ; 
	Params->Cde = 1 ; 
	Params->Tde = 35 ;
	// Sigmoid : 
	Params->Csig = 0.5 ;
	// Params->Csig = 80000 ; // control the gradient of sigmoid ,   sigmoid : 5W ~ 25W -> 0 ~ 1
	// Params->Csig2 = 230000 ; // control right shift of sigmoid 

	Params->Tffi = 8 ; ////////////////////////////////////////////////////////////////////////////////////////////
	
	Params->Kspi = 4 ; // 10 ; // scale parameter
	Params->TspiF = 50 ; // threshold for spike_Frequency

	Params->Delta = 0.01 ;
	Params->Tsfa = 0.003 ;
	
	Params->Tspi = 0.86 ; // threshold for spike///////////////////////////////////////////////////////////////////

	Params->W_base_on = 1 ;
	Params->W_base_off = 0.3 ;
	Params->W_on_bias = 1 ;
	Params->W_off_bias = 0.3 ;


	printf("LGMD-Basic Params Initialization Over......\r\n") ;

	return 0 ;
}

float sigmoid(float MP , float Csig , float Csig2) 
{
	// *SMP = 1/(1 + exp(-(*MP+70000)*(1/(N_cell*Csig) ) ) ) ;
	float kf = MP , K ;
	float C = Csig ; // control the gradient of Sigmoid_Fouction
	float C2 = Csig2 ; // control right shift of Sigmoid_Fouction

	K = 1/(1 + exp(-(1/C)*(kf - C2 ) ) ) ;

	return K ;
}

uint8_t LGMD_Calculating(LGMD_pControlTypedef* hLGMD)
{
	// Initialization :

	uint8_t image_opt_cur[3] = {2,0,1} ;
	uint8_t image_opt_pre[3] = {1,2,0} ;

	hLGMD->currentImage = (*(hLGMD->hFrameCount))%3 ;
	hLGMD->currentDiffImage = !(hLGMD->currentDiffImage) ; // Alter between "pre and cur"

	uint16_t* image_pixel_cur ;	// point to current image's pixel
	uint16_t* image_pixel_pre ;	// point to previous image's pixel
	int8_t* P_Layer_Pixel ;	// point to current P_Layer's pixel
	uint8_t *ON_pixel_cur , *ON_pixel_pre , *OFF_pixel_cur , *OFF_pixel_pre ;

	// L1 L2 pixel
	image_pixel_cur=(uint16_t*)(((uint8_t*)&Image[image_opt_cur[(*(hLGMD->hFrameCount))%3]][0][0])+1);
	image_pixel_pre=(uint16_t*)(((uint8_t*)&Image[image_opt_pre[(*(hLGMD->hFrameCount))%3]][0][0])+1);
	//image_pixel_cur = &Image[image_opt_cur[hLGMD->currentImage] ][0][0] ; // uint 16
	//image_pixel_pre = &Image[image_opt_pre[hLGMD->currentImage] ][0][0] ; // uint 16
	// P pixel
	P_Layer_Pixel = &Diff_Image[hLGMD->currentDiffImage][0][0] ;	// uint 8
	// ON OFF pixel
	ON_pixel_cur = &hLGMD->Model->Layers.ON[hLGMD->currentDiffImage][0][0] ;
	ON_pixel_pre = &hLGMD->Model->Layers.ON[!(hLGMD->currentDiffImage)][0][0] ;
	OFF_pixel_cur = &hLGMD->Model->Layers.OFF[hLGMD->currentDiffImage][0][0] ;
	OFF_pixel_pre = &hLGMD->Model->Layers.OFF[!(hLGMD->currentDiffImage)][0][0] ;


	// P_Layer (E_Layer) :
	// P range : -127 ~ 127
	uint16_t N_cell = Image_Height*Image_Width , n ;
	int16_t Diff ;
	float alpha_left_on , alpha_left_off ;
	float sum_ffi_on = 0 , sum_ffi_off = 0 , ave_ffi = 0 , alpha_ffi_on , alpha_ffi_off , clip ;
	float *pre_ffi_on , *cur_ffi_on , *cur_ffi_off , *pre_ffi_off ;
	clip = hLGMD->Model->Params.clip ;
	float TH_on , TH_off ;

	// alpha_ffi = hLGMD->Model->Params.alpha_ffi ;
	// pre_ffi = &hLGMD->Model->Results.FFI[!hLGMD->currentDiffImage] ;
	// cur_ffi = &hLGMD->Model->Results.FFI[hLGMD->currentDiffImage] ;	
	/**********************************************************************************************/
	alpha_left_on = hLGMD->Model->Params.alpha_left_on ;
	alpha_left_off = hLGMD->Model->Params.alpha_left_off ;
	
	alpha_ffi_on = hLGMD->Model->Params.alpha_ffi_on ;
	alpha_ffi_off = hLGMD->Model->Params.alpha_ffi_off ;

	cur_ffi_on = &hLGMD->Model->Results.FFI_ON[hLGMD->currentDiffImage] ;
	pre_ffi_on = &hLGMD->Model->Results.FFI_ON[!hLGMD->currentDiffImage] ;
	cur_ffi_off = &hLGMD->Model->Results.FFI_OFF[hLGMD->currentDiffImage] ;
	pre_ffi_off = &hLGMD->Model->Results.FFI_OFF[!hLGMD->currentDiffImage] ;

	TH_on = hLGMD->Model->Params.TH_on ;
	TH_off = hLGMD->Model->Params.TH_off ;
	/**********************************************************************************************/
	

	// P and ON_OFF :
	for( n = 0 ; n < N_cell ; n ++ )
	{
		Diff = (*image_pixel_cur - *image_pixel_pre)>>9 ; //high-pass in photoreceptors

		// allocation(-127 ~ 127) :
		if( Diff > 127){
			*P_Layer_Pixel = 127 ;
		}else if( Diff < - 127 ){
			*P_Layer_Pixel = -127 ;
		}else{
			*P_Layer_Pixel = Diff ;
		}
		
		// ON_OFF :
		if(*P_Layer_Pixel >= clip ){
			*ON_pixel_cur = (uint8_t)(*P_Layer_Pixel + alpha_left_on*(*ON_pixel_pre) ) ;
			*OFF_pixel_cur = (uint8_t)(0 + alpha_left_off*(*OFF_pixel_pre) ) ;
			// FFI : 
			sum_ffi_on += *(ON_pixel_cur) ;
		}else{
			*OFF_pixel_cur = (uint8_t)(abs(*P_Layer_Pixel) + alpha_left_off*(*OFF_pixel_pre) ) ;
			*ON_pixel_cur = (uint8_t)(0 + alpha_left_on*(*ON_pixel_pre) ) ;
			// FFI :
			sum_ffi_off += *(OFF_pixel_cur) ;
		}

		// FFI calculate :
		// sum_ffi += *(P_Layer_Pixel) ;
		

		// move to next Pixel :
		image_pixel_cur ++ ; // L2_pixel ++ 
		image_pixel_pre ++ ; // L1_pixel ++ 
		P_Layer_Pixel ++ ; // P_pixel ++
		ON_pixel_cur ++ ; // ON_pixel ++
		OFF_pixel_cur ++ ; // OFF_pixel ++
	}

	// FFI_ON_OFF : 
	*cur_ffi_on = sum_ffi_on / N_cell ;
	*cur_ffi_off = sum_ffi_off / N_cell ;
	// FFI = (alpha)*cur_FFI + (1-alpha)*pre_FFI , alpha = tau_1/(tau_1+tau_in) : 
	*cur_ffi_on = alpha_ffi_on*(*cur_ffi_on) + (1-alpha_ffi_on)*(*pre_ffi_on) ;
	*cur_ffi_off = alpha_ffi_off*(*cur_ffi_off) + (1-alpha_ffi_off)*(*pre_ffi_off) ;



	// watch : 
	float *cur_ave_ffi_ON_watch , *cur_ave_ffi_OFF_watch ;
	cur_ave_ffi_ON_watch = &hLGMD->Model->Results.cur_ave_ffi_ON ;
	cur_ave_ffi_OFF_watch = &hLGMD->Model->Results.cur_ave_ffi_OFF ;
	*cur_ave_ffi_ON_watch = *cur_ffi_on ;
	*cur_ave_ffi_OFF_watch = *cur_ffi_off ;




	/* FFI in previous Model :
	// FFI in normal ON_OFF :
	*cur_ffi = sum_ffi / N_cell ;
	*cur_ffi = alpha_ffi*(*cur_ffi ) + (1-alpha_ffi )*(*pre_ffi ) ; // FFI = (alpha)*cur_FFI + (1-alpha)*pre_FFI , alpha = tau_1/(tau_1+tau_in)
	

	// FFI in LGMD-Basic : 
	ave_ffi = fabs(sum_ffi/N_cell) ;
	if( ave_ffi < Tffi ) {
		*cur_ffi = 0 ;
	}else{
		*cur_ffi = 1 ;
	}
	*/

	// hLGMD->Model->Results.sum_ffi = sum_ffi ;  // experiment

	if((*cur_ffi_on<TH_on) && (*cur_ffi_off<TH_off) ){
		S_Calculate_none(hLGMD) ;
	}else if((*cur_ffi_on<TH_on) && (*cur_ffi_off>=TH_off) ){
		S_Calculate_OFF(hLGMD) ;
	}else if((*cur_ffi_on>=TH_on) && (*cur_ffi_off<TH_off) ){
		S_Calculate_ON(hLGMD) ;
	}else{
		S_Calculate_ON_and_OFF(hLGMD) ;
	}
	// in Founction "LGMD_Calculating(LGMD_pControlTypedef* hLGMD)" , the global argument "*hLGMD" was transtated to local argument.
	// so at inner "LGMD_Calculating(LGMD_pControlTypedef* hLGMD)" founction , "hLGMD" means a pointer but not a normal argument as global argument initailly set.


/*
	// S :  

	LGMDstruct_Layer* Layers = &hLGMD->Model->Layers ;
	LGMDstruct_Params* Params = &hLGMD->Model->Params ;
	LGMDstruct_Result* Result = &hLGMD->Model->Results ;
	uint8_t *S_Layer_Pixel , *G_Layer_Pixel ;

	// Delay_params for I convolution : 
	uint8_t i , j ;
	uint16_t self , adjacent , diagonal , pre_self , pre_adjacent , pre_diagnal ;
	float alpha_on_self , alpha_on_adj , alpha_on_diag , alpha_off_self , alpha_off_adj , alpha_off_diag ;	uint16_t I_on , I_off ;
	float S_on , S_off ;

	alpha_on_self = Params->alpha_on_self ;
	alpha_on_adj = Params->alpha_on_adj ;
	alpha_on_diag = Params->alpha_on_diag ;
	alpha_off_self = Params->alpha_off_self ;
	alpha_off_adj = Params->alpha_off_adj ;
	alpha_off_diag = Params->alpha_off_diag ;

	// Wi coefficient , Bias by FFI :
	float Tffi , W_base ; 
	float *W_on_bias ,* W_off_bias ;
	float *cur_ffi_on , *cur_ffi_off ;
	
	cur_ffi_on = &hLGMD->Model->Results.FFI_ON[hLGMD->currentDiffImage] ;
	cur_ffi_off = &hLGMD->Model->Results.FFI_OFF[hLGMD->currentDiffImage] ;

	Tffi = hLGMD->Model->Params.Tffi ;
	*W_on_bias = &hLGMD->Model->Params.W_on_bias ;
	*W_off_bias = &hLGMD->Model->Params.W_off_bias ;
	W_base = hLGMD->Model->Params.W_base ;

	*W_on_bias = ((*cur_ffi_on/Tffi )>W_base ) ? (*cur_ffi_on/Tffi ) : (W_base ) ;
	*W_off_bias = ((*cur_ffi_off/Tffi )>W_base ) ? (*cur_ffi_off/Tffi ) : (W_base ) ;

	// superliner cumputation : 
	float theta_1 , theta_2 , theta_3 ;
	theta_1 = Params->theta_1 ;
	theta_2 = Params->theta_2 ;
	theta_3 = Params->theta_3 ;

	
	for(i = 1 ; i < Image_Height - 1 ; i ++ )
	{
		// core
		// To use ptr calculation in the second_fold_for_loop :
		P_Layer_Pixel = &Diff_Image[hLGMD->currentDiffImage][i][0] ;

		ON_pixel_cur = &Layers->ON[hLGMD->currentDiffImage][i][0] ;
		ON_pixel_pre = &Layers->ON[!(hLGMD->currentDiffImage)][i][0] ;

		OFF_pixel_cur = &Layers->OFF[hLGMD->currentDiffImage][i][0] ;
		OFF_pixel_pre = &Layers->OFF[!(hLGMD->currentDiffImage)][i][0] ;

		// I_Layer_Pixel = &hLGMD->Model->Layers.I[i][0] ;
		S_Layer_Pixel = &hLGMD->Model->Layers.S[i][0] ;
		G_Layer_Pixel = &hLGMD->Model->Layers.G[i][0] ;		

		for(j = 1 ; j < Image_Width - 1 ; j ++ )
		{
			// I_ON : convolving delayed excitation in ON channels to form ON inhibition
			self = *(ON_pixel_cur+j) ;

			adjacent = (*(ON_pixel_cur+j-1)>>2)
					 + (*(ON_pixel_cur+j+1)>>2)
					 + (*(ON_pixel_cur+j-Image_Width)>>2)
					 + (*(ON_pixel_cur+j+Image_Width)>>2) ;
			
			diagonal = (*(ON_pixel_cur+j-Image_Width-1)>>2)
					 + (*(ON_pixel_cur+j-Image_Width+1)>>2)
					 + (*(ON_pixel_cur+j+Image_Width-1)>>2)
					 + (*(ON_pixel_cur+j+Image_Width+1)>>2) ;

			pre_self = *(ON_pixel_pre+j) ;

			pre_adjacent = (*(ON_pixel_pre+j-1)>>2)
						 + (*(ON_pixel_pre+j+1)>>2)
						 + (*(ON_pixel_pre+j-Image_Width)>>2)
						 + (*(ON_pixel_pre+j+Image_Width)>>2) ;
			
			pre_diagnal  = (*(ON_pixel_pre+j-Image_Width-1)>>2)
						 + (*(ON_pixel_pre+j-Image_Width+1)>>2)
						 + (*(ON_pixel_pre+j+Image_Width-1)>>2)
						 + (*(ON_pixel_pre+j+Image_Width+1)>>2) ;
			
			I_on = (uint16_t )(alpha_on_self)*self     - (1-alpha_on_self)*(pre_self)
							+ (alpha_on_adj )*adjacent - (1-alpha_on_adj)*(pre_adjacent)
							+ (alpha_on_diag)*diagonal - (1-alpha_on_diag)*(pre_diagnal) ;


			// I_OFF : convolving delayed excitation in OFF channels to form OFF inhibition
			self = *(OFF_pixel_cur+j) ;

			adjacent = (*(OFF_pixel_cur+j-1)>>2)
					 + (*(OFF_pixel_cur+j+1)>>2)
					 + (*(OFF_pixel_cur+j-Image_Width)>>2)
					 + (*(OFF_pixel_cur+j+Image_Width)>>2) ;
			
			diagonal = (*(OFF_pixel_cur+j-Image_Width-1)>>2)
					 + (*(OFF_pixel_cur+j-Image_Width+1)>>2)
					 + (*(OFF_pixel_cur+j+Image_Width-1)>>2)
					 + (*(OFF_pixel_cur+j+Image_Width+1)>>2) ;

			pre_self = *(OFF_pixel_pre+j) ;

			pre_adjacent = (*(OFF_pixel_pre+j-1)>>2)
						 + (*(OFF_pixel_pre+j+1)>>2)
						 + (*(OFF_pixel_pre+j-Image_Width)>>2)
						 + (*(OFF_pixel_pre+j+Image_Width)>>2) ;
			
			pre_diagnal  = (*(OFF_pixel_pre+j-Image_Width-1)>>2)
						 + (*(OFF_pixel_pre+j-Image_Width+1)>>2)
						 + (*(OFF_pixel_pre+j+Image_Width-1)>>2)
						 + (*(OFF_pixel_pre+j+Image_Width+1)>>2) ;
			
			I_off = (uint16_t )(alpha_off_self)*self    - (1-alpha_off_self)*(pre_self)
							+ (alpha_off_adj )*adjacent - (1-alpha_off_adj)*(pre_adjacent)
							+ (alpha_off_diag)*diagonal - (1-alpha_off_diag)*(pre_diagnal) ;

			// S : 
			S_on  = *(ON_pixel_cur+j) - (*W_on_bias)*I_on ;
			S_off = *(ON_pixel_cur+j) - (*W_on_bias)*I_off ;
			if(S_on<0) S_on=0 ;
			if(S_off<0) S_off=0 ;
			
			// superliner cumputation:
			*(S_Layer_Pixel+j) = (uint8_t)(theta_1*S_on + theta_2*S_off + theta_3*S_on*S_off ) ;

			// // LGMD-Basic : 
			//self = 0 ; 
			//adjacent = (*(P_Layer_Pixel+j-1)>>2)//adjacent overflow??????
			//		 + (*(P_Layer_Pixel+j+1)>>2)
			//		 + (*(P_Layer_Pixel+j-Image_Width)>>2)
			//		 + (*(P_Layer_Pixel+j+Image_Width)>>2) ; // adjacent_weight = 1/4
			//
			//diagonal = (*(P_Layer_Pixel+j-Image_Width-1)>>3)
			//		 + (*(P_Layer_Pixel+j-Image_Width+1)>>3)
			//		 + (*(P_Layer_Pixel+j+Image_Width-1)>>3)
			//		 + (*(P_Layer_Pixel+j+Image_Width+1)>>3) ; // diagnal_weight = 1/8
			// I cell :
			//*(I_Layer_Pixel+j) =  diagonal + adjacent + 0 ; // self_wweight = 0

			// S cell :
			//if( (*(I_Layer_Pixel+j)) * (*(P_Layer_Pixel+j)) > 0 ){
			//	*(S_Layer_Pixel+j) = *(P_Layer_Pixel+j) - Wi*(*(I_Layer_Pixel+j)) ; 
			//}else{
			//	*(S_Layer_Pixel) = 0 ;
			//}
			//
			

			// Ce : need completed space information of S_Layer !
		}
	} // now we have completed S_Layer.
*/



	// G_Layer :
	LGMDstruct_Layer* Layers = &hLGMD->Model->Layers ;
	LGMDstruct_Params* Params = &hLGMD->Model->Params ;
	LGMDstruct_Result* Result = &hLGMD->Model->Results ;
	uint8_t *S_Layer_Pixel , *G_Layer_Pixel ;
	uint8_t* Ce_Pixel ;
	uint8_t Max_Ce = 0 , i , j ;
	float w = 0 , delta , Cw ;
	delta = hLGMD->Model->Params.Delta ;
	Cw = hLGMD->Model->Params.Cw ;
	// Ce : (Temporally be put into G_Layer_matrix )
	for(i = 1 ; i < Image_Height - 1 ; i ++ )
	{
		S_Layer_Pixel = &hLGMD->Model->Layers.S[i][0] ;
		Ce_Pixel = &hLGMD->Model->Layers.G[i][0] ;
		
		for(j = 1 ; j < Image_Width - 1 ; j ++ )
		{
			*(Ce_Pixel+j) = ( (*(S_Layer_Pixel+j)) // self
					        + (*(S_Layer_Pixel+j-1))
					        + (*(S_Layer_Pixel+j+1))
					        + (*(S_Layer_Pixel+j-Image_Width))
					        + (*(S_Layer_Pixel+j+Image_Width)) // adjecent
					        + (*(S_Layer_Pixel-Image_Width-1))
					        + (*(S_Layer_Pixel-Image_Width+1))
					        + (*(S_Layer_Pixel+Image_Width-1))
					        + (*(S_Layer_Pixel+Image_Width+1)) )/9 ; // diagnal 

			if( *(Ce_Pixel+j) > Max_Ce )
				Max_Ce = *(Ce_Pixel+j) ;
		}
	}
	w = delta + ((float)Max_Ce)/Cw ;
	// hLGMD->watch1 = w ; // experiment


	// G and G~ :   
	float *MP ;
	float Cde , Tde , Csig , Tspi ;
	MP = &hLGMD->Model->Results.MP ;
	*MP = 0 ;
	
	Cde = hLGMD->Model->Params.Cde ;
	Tde = hLGMD->Model->Params.Tde ;
	Csig = hLGMD->Model->Params.Csig ;
	Tspi = hLGMD->Model->Params.Tspi ;
	/*
	SMP = &hLGMD->Model->Results.SMP ; // after Sigmoid
	uint8_t *Constant_Spike , *Cfinal , *Collision ;
	Nsp = hLGMD->Model->Params.Nsp ;
	Constant_Spike = &hLGMD->Model->Results.Constant_spike ;
	Cfinal = &hLGMD->Model->Results.Cfinal ;
	Collision = &hLGMD->Model->Results.Collision ;
	*/

	// G : ( NO need spatial information of Ce_Layer , so we do not need a seperated Ce_Layer ! )
	for(i = 1 ; i < Image_Height - 1 ; i ++ )
	{
		S_Layer_Pixel = &hLGMD->Model->Layers.S[i][0] ;
		Ce_Pixel = &hLGMD->Model->Layers.G[i][0] ;
		G_Layer_Pixel = &hLGMD->Model->Layers.G[i][0] ;

		for(j = 1 ; j < Image_Width - 1 ; j ++ )
		{
			*(G_Layer_Pixel+j) = (uint8_t)((*(S_Layer_Pixel+j)) * (*(Ce_Pixel+j))/w) ;
			// G~ : 
			if( (*(G_Layer_Pixel+j))*Cde < Tde )
				*(G_Layer_Pixel+j) = 0 ;
				
			// MP (Kf) :
			*MP += *(G_Layer_Pixel+j) ;
		}
	}

	// Menbrence_Potential , SFA_mechanism :
	float alpha_sfa , Tsfa ;
	float *cur_SMP , *pre_SMP , diff_SMP , *cur_LGMD_out , *pre_LGMD_out ;

	alpha_sfa = Params->alpha_sfa ;
	Tsfa = Params->Tsfa ;
	

	cur_SMP = &Result->SMP[hLGMD->currentDiffImage] ;
	pre_SMP = &Result->SMP[!(hLGMD->currentDiffImage)] ;
	cur_LGMD_out = &Result->LGMD_output[hLGMD->currentDiffImage] ;
	pre_LGMD_out = &Result->LGMD_output[!(hLGMD->currentDiffImage)] ;
	
	// Sigmoid : 
	 *cur_SMP = 1/(1 + exp(-(*MP)/(N_cell*Csig) ) ) ;
	// *cur_SMP = sigmoid( *MP , hLGMD->Model->Params.Csig , hLGMD->Model->Params.Csig2 ) ;

	// SFA.
	diff_SMP = (*cur_SMP) - (*pre_SMP) ;
	if(diff_SMP <= Tsfa ){
		*cur_LGMD_out = alpha_sfa*(*pre_LGMD_out + diff_SMP );
	}else{
		*cur_LGMD_out = alpha_sfa*(*cur_SMP );
	} 
	
	//if (*cur_LGMD_out < 0.5 )
	//	*cur_LGMD_out = 0.5 ;

	// Spiking_mechanism :
	float Kspi , TspiF ;
	uint8_t *Spike , *constant;
	uint8_t frame = (*(hLGMD->hFrameCount ) ) % 10 ;
	//float *Spike_Frequency ;
	//Spike_Frequency = &Result->Spike_Frequency ;
	uint8_t *Collision ; //  0 / 1 , FINAL

    Kspi = Params->Kspi ;
	Tspi = Params->Tspi ;
	TspiF = Params->TspiF ;
	Spike = &Result->Spike ;
	constant = &Result->constant ;
	Collision = &Result->Collision ;
	//*Spike_Frequency = 0 ;
	*Collision = 0 ;

	*Spike = floor(exp(Kspi*(*cur_LGMD_out - Tspi ) ) ) ;
	/*
	// Spike Frequency :
	(*Spike_Frequency) = Spike[0] + Spike[1] + Spike[2] +
	                     Spike[3] + Spike[4] + Spike[5] +
	                     Spike[6] + Spike[7] + Spike[8] +
	                     Spike[9] ;
	
	for( i = 0 ; i < 10 ; i ++ )
	{
		(*Spike_Frequency) += (float)(*(Spike+i ) ) ;
	}
	*Spike_Frequency = (*Spike_Frequency)*100 / (1000/hLGMD->processRate ) ; // calculate frequency
	
	*/


	// Collision :

	//*Collision = (*Spike_Frequency >= TspiF) ? 1 : 0 ;
	if(*Spike > 0 )
	{
		(*constant) ++ ;
		if(*constant >= 4){
			*Collision = 1 ;
			*constant = 0 ;
		}
	}else{
		*constant = 0 ;
	}


	/*  LGMD-Basic spike mechanism : 
	// Spike Number :
	if( *SMP > Tspi ){
		(*Constant_Spike) ++ ;
	}else{
		*Constant_Spike = 0 ; // corrupt !!
	}

	// Cfinal :
	if( *Constant_Spike >= 4 ){
		*Cfinal = 1 ;
	}else{
		*Cfinal = 0 ;
	}

	// Collision :
	if( *Cfinal && !(*pre_ffi) ){
		*Collision = 1 ;
	}else{
		*Collision = 0 ;
	}
	*/

	return 0 ;
}


// S_Layer with both ON_OFF
void S_Calculate_ON_and_OFF(LGMD_pControlTypedef* hLGMD) 
{
	// S :  

	LGMDstruct_Layer* Layers = &hLGMD->Model->Layers ;
	LGMDstruct_Params* Params = &hLGMD->Model->Params ;
	LGMDstruct_Result* Result = &hLGMD->Model->Results ;
	uint8_t *S_Layer_Pixel , *G_Layer_Pixel ;

	int8_t* P_Layer_Pixel ;	// point to current P_Layer's pixel
	uint8_t *ON_pixel_cur , *ON_pixel_pre , *OFF_pixel_cur , *OFF_pixel_pre ;
	// P pixel
	P_Layer_Pixel = &Diff_Image[hLGMD->currentDiffImage][0][0] ;	// uint 8
	// ON OFF pixel
	ON_pixel_cur = &hLGMD->Model->Layers.ON[hLGMD->currentDiffImage][0][0] ;
	ON_pixel_pre = &hLGMD->Model->Layers.ON[!(hLGMD->currentDiffImage)][0][0] ;
	OFF_pixel_cur = &hLGMD->Model->Layers.OFF[hLGMD->currentDiffImage][0][0] ;
	OFF_pixel_pre = &hLGMD->Model->Layers.OFF[!(hLGMD->currentDiffImage)][0][0] ;

	// Delay_params for I convolution : 
	uint8_t i , j ;
	uint16_t self , adjacent , diagonal , pre_self , pre_adjacent , pre_diagnal ;
	float alpha_on_self , alpha_on_adj , alpha_on_diag , alpha_off_self , alpha_off_adj , alpha_off_diag ;	
	uint16_t I_on , I_off ;
	float S_on , S_off ;

	alpha_on_self = Params->alpha_on_self ;
	alpha_on_adj = Params->alpha_on_adj ;
	alpha_on_diag = Params->alpha_on_diag ;
	alpha_off_self = Params->alpha_off_self ;
	alpha_off_adj = Params->alpha_off_adj ;
	alpha_off_diag = Params->alpha_off_diag ;

	// Wi coefficient , Bias by FFI :
	float Tffi , W_base_on , W_base_off ; 
	float *W_on_bias ,* W_off_bias ;
	float *cur_ffi_on , *cur_ffi_off ;
	
	cur_ffi_on = &hLGMD->Model->Results.FFI_ON[hLGMD->currentDiffImage] ;
	cur_ffi_off = &hLGMD->Model->Results.FFI_OFF[hLGMD->currentDiffImage] ;

	Tffi = hLGMD->Model->Params.Tffi ;
	W_on_bias = &hLGMD->Model->Params.W_on_bias ;
	W_off_bias = &hLGMD->Model->Params.W_off_bias ;
	W_base_on = hLGMD->Model->Params.W_base_on ;
	W_base_off = hLGMD->Model->Params.W_base_off ;

	*W_on_bias = ((*cur_ffi_on/Tffi )>W_base_on ) ? (*cur_ffi_on/Tffi ) : (W_base_on ) ;
	*W_off_bias = ((*cur_ffi_off/Tffi )>W_base_off ) ? (*cur_ffi_off/Tffi ) : (W_base_off ) ;

	// superliner cumputation : 
	float theta_1 , theta_2 , theta_3 ;
	theta_1 = Params->theta_1 ;
	theta_2 = Params->theta_2 ;
	theta_3 = Params->theta_3 ;

	
	for(i = 1 ; i < Image_Height - 1 ; i ++ )
	{
		// core
		// To use ptr calculation in the second_fold_for_loop :
		P_Layer_Pixel = &Diff_Image[hLGMD->currentDiffImage][i][0] ;

		ON_pixel_cur = &Layers->ON[hLGMD->currentDiffImage][i][0] ;
		ON_pixel_pre = &Layers->ON[!(hLGMD->currentDiffImage)][i][0] ;

		OFF_pixel_cur = &Layers->OFF[hLGMD->currentDiffImage][i][0] ;
		OFF_pixel_pre = &Layers->OFF[!(hLGMD->currentDiffImage)][i][0] ;

		// I_Layer_Pixel = &hLGMD->Model->Layers.I[i][0] ;
		S_Layer_Pixel = &hLGMD->Model->Layers.S[i][0] ;
		G_Layer_Pixel = &hLGMD->Model->Layers.G[i][0] ;		

		for(j = 1 ; j < Image_Width - 1 ; j ++ )
		{
			// I_ON : convolving delayed excitation in ON channels to form ON inhibition
			self = *(ON_pixel_cur+j) ;

			adjacent = (*(ON_pixel_cur+j-1)>>2)
					 + (*(ON_pixel_cur+j+1)>>2)
					 + (*(ON_pixel_cur+j-Image_Width)>>2)
					 + (*(ON_pixel_cur+j+Image_Width)>>2) ;
			
			diagonal = (*(ON_pixel_cur+j-Image_Width-1)>>3)
					 + (*(ON_pixel_cur+j-Image_Width+1)>>3)
					 + (*(ON_pixel_cur+j+Image_Width-1)>>3)
					 + (*(ON_pixel_cur+j+Image_Width+1)>>3) ;

			pre_self = *(ON_pixel_pre+j) ;

			pre_adjacent = (*(ON_pixel_pre+j-1)>>2)
						 + (*(ON_pixel_pre+j+1)>>2)
						 + (*(ON_pixel_pre+j-Image_Width)>>2)
						 + (*(ON_pixel_pre+j+Image_Width)>>2) ;
			
			pre_diagnal  = (*(ON_pixel_pre+j-Image_Width-1)>>3)
						 + (*(ON_pixel_pre+j-Image_Width+1)>>3)
						 + (*(ON_pixel_pre+j+Image_Width-1)>>3)
						 + (*(ON_pixel_pre+j+Image_Width+1)>>3) ;
			
			I_on = (uint16_t )(alpha_on_self)*self     + (1-alpha_on_self)*(pre_self)
							+ (alpha_on_adj )*adjacent + (1-alpha_on_adj)*(pre_adjacent)
							+ (alpha_on_diag)*diagonal + (1-alpha_on_diag)*(pre_diagnal) ;


			// I_OFF : convolving delayed excitation in OFF channels to form OFF inhibition
			self = *(OFF_pixel_cur+j) ;

			adjacent = (*(OFF_pixel_cur+j-1)>>2)
					 + (*(OFF_pixel_cur+j+1)>>2)
					 + (*(OFF_pixel_cur+j-Image_Width)>>2)
					 + (*(OFF_pixel_cur+j+Image_Width)>>2) ;
			
			diagonal = (*(OFF_pixel_cur+j-Image_Width-1)>>3)
					 + (*(OFF_pixel_cur+j-Image_Width+1)>>3)
					 + (*(OFF_pixel_cur+j+Image_Width-1)>>3)
					 + (*(OFF_pixel_cur+j+Image_Width+1)>>3) ;

			pre_self = *(OFF_pixel_pre+j) ;

			pre_adjacent = (*(OFF_pixel_pre+j-1)>>2)
						 + (*(OFF_pixel_pre+j+1)>>2)
						 + (*(OFF_pixel_pre+j-Image_Width)>>2)
						 + (*(OFF_pixel_pre+j+Image_Width)>>2) ;
			
			pre_diagnal  = (*(OFF_pixel_pre+j-Image_Width-1)>>3)
						 + (*(OFF_pixel_pre+j-Image_Width+1)>>3)
						 + (*(OFF_pixel_pre+j+Image_Width-1)>>3)
						 + (*(OFF_pixel_pre+j+Image_Width+1)>>3) ;
			
			I_off = (uint16_t )(alpha_off_self)*self    + (1-alpha_off_self)*(pre_self)
							+ (alpha_off_adj )*adjacent + (1-alpha_off_adj)*(pre_adjacent)
							+ (alpha_off_diag)*diagonal + (1-alpha_off_diag)*(pre_diagnal) ;

			// S : 
			S_on  = *(ON_pixel_cur+j) - (*W_on_bias)*I_on ;
			S_off = *(ON_pixel_cur+j) - (*W_on_bias)*I_off ;
			if(S_on<0) S_on=0 ;
			if(S_off<0) S_off=0 ;
			
			// superliner cumputation:
			*(S_Layer_Pixel+j) = (uint8_t)(theta_1*S_on + theta_2*S_off + theta_3*S_on*S_off ) ;

			/* // LGMD-Basic : 
			self = 0 ; 
			adjacent = (*(P_Layer_Pixel+j-1)>>2)//adjacent overflow??????
					 + (*(P_Layer_Pixel+j+1)>>2)
					 + (*(P_Layer_Pixel+j-Image_Width)>>2)
					 + (*(P_Layer_Pixel+j+Image_Width)>>2) ; // adjacent_weight = 1/4

			diagonal = (*(P_Layer_Pixel+j-Image_Width-1)>>3)
					 + (*(P_Layer_Pixel+j-Image_Width+1)>>3)
					 + (*(P_Layer_Pixel+j+Image_Width-1)>>3)
					 + (*(P_Layer_Pixel+j+Image_Width+1)>>3) ; // diagnal_weight = 1/8
			// I cell :
			*(I_Layer_Pixel+j) =  diagonal + adjacent + 0 ; // self_wweight = 0

			// S cell :
			if( (*(I_Layer_Pixel+j)) * (*(P_Layer_Pixel+j)) > 0 ){
				*(S_Layer_Pixel+j) = *(P_Layer_Pixel+j) - Wi*(*(I_Layer_Pixel+j)) ; 
			}else{
				*(S_Layer_Pixel) = 0 ;
			}
			*/
			

			// Ce : need completed space information of S_Layer !
		}
	} // now we have completed S_Layer.
}

// S_Layer with only ON_Channel
void S_Calculate_ON(LGMD_pControlTypedef* hLGMD) 
{
	// S :  

	LGMDstruct_Layer* Layers = &hLGMD->Model->Layers ;
	LGMDstruct_Params* Params = &hLGMD->Model->Params ;
	LGMDstruct_Result* Result = &hLGMD->Model->Results ;
	uint8_t *S_Layer_Pixel ;

	int8_t* P_Layer_Pixel ;	// point to current P_Layer's pixel
	uint8_t *ON_pixel_cur , *ON_pixel_pre , *OFF_pixel_cur , *OFF_pixel_pre ;
	// P pixel
	P_Layer_Pixel = &Diff_Image[hLGMD->currentDiffImage][0][0] ;	// uint 8
	// ON OFF pixel
	ON_pixel_cur = &hLGMD->Model->Layers.ON[hLGMD->currentDiffImage][0][0] ;
	ON_pixel_pre = &hLGMD->Model->Layers.ON[!(hLGMD->currentDiffImage)][0][0] ;
	// OFF_pixel_cur = &hLGMD->Model->Layers.OFF[hLGMD->currentDiffImage][0][0] ;
	// OFF_pixel_pre = &hLGMD->Model->Layers.OFF[!(hLGMD->currentDiffImage)][0][0] ;

	// Delay_params for I convolution : 
	uint8_t i , j ;
	uint16_t self , adjacent , diagonal , pre_self , pre_adjacent , pre_diagnal ;
	float alpha_on_self , alpha_on_adj , alpha_on_diag , alpha_off_self , alpha_off_adj , alpha_off_diag ;	
	uint16_t I_on , I_off ;
	float S_on , S_off ;

	alpha_on_self = Params->alpha_on_self ;
	alpha_on_adj = Params->alpha_on_adj ;
	alpha_on_diag = Params->alpha_on_diag ;
	// alpha_off_self = Params->alpha_off_self ;
	// alpha_off_adj = Params->alpha_off_adj ;
	// alpha_off_diag = Params->alpha_off_diag ;

	// Wi coefficient , Bias by FFI :
	float Tffi , W_base_on ; 
	float *W_on_bias ,* W_off_bias ;
	float *cur_ffi_on , *cur_ffi_off ;
	
	cur_ffi_on = &hLGMD->Model->Results.FFI_ON[hLGMD->currentDiffImage] ;
	// cur_ffi_off = &hLGMD->Model->Results.FFI_OFF[hLGMD->currentDiffImage] ;

	Tffi = hLGMD->Model->Params.Tffi ;
	W_on_bias = &hLGMD->Model->Params.W_on_bias ;
	// *W_off_bias = &hLGMD->Model->Params.W_off_bias ;
	W_base_on = hLGMD->Model->Params.W_base_on ;

	*W_on_bias = ((*cur_ffi_on/Tffi )>W_base_on ) ? (*cur_ffi_on/Tffi ) : (W_base_on ) ;
	// *W_off_bias = ((*cur_ffi_off/Tffi )>W_base ) ? (*cur_ffi_off/Tffi ) : (W_base ) ;

	// superliner cumputation : 
	float theta_1 , theta_2 , theta_3 ;
	theta_1 = Params->theta_1 ;
	theta_2 = Params->theta_2 ;
	theta_3 = Params->theta_3 ;

	
	for(i = 1 ; i < Image_Height - 1 ; i ++ )
	{
		// core
		// To use ptr calculation in the second_fold_for_loop :
		P_Layer_Pixel = &Diff_Image[hLGMD->currentDiffImage][i][0] ;

		ON_pixel_cur = &Layers->ON[hLGMD->currentDiffImage][i][0] ;
		ON_pixel_pre = &Layers->ON[!(hLGMD->currentDiffImage)][i][0] ;

		OFF_pixel_cur = &Layers->OFF[hLGMD->currentDiffImage][i][0] ;
		OFF_pixel_pre = &Layers->OFF[!(hLGMD->currentDiffImage)][i][0] ;

		// I_Layer_Pixel = &hLGMD->Model->Layers.I[i][0] ;
		S_Layer_Pixel = &hLGMD->Model->Layers.S[i][0] ;
		// G_Layer_Pixel = &hLGMD->Model->Layers.G[i][0] ;		

		for(j = 1 ; j < Image_Width - 1 ; j ++ )
		{
			// I_ON : convolving delayed excitation in ON channels to form ON inhibition
			self = *(ON_pixel_cur+j) ;

			adjacent = (*(ON_pixel_cur+j-1)>>2)
					 + (*(ON_pixel_cur+j+1)>>2)
					 + (*(ON_pixel_cur+j-Image_Width)>>2)
					 + (*(ON_pixel_cur+j+Image_Width)>>2) ;
			
			diagonal = (*(ON_pixel_cur+j-Image_Width-1)>>3)
					 + (*(ON_pixel_cur+j-Image_Width+1)>>3)
					 + (*(ON_pixel_cur+j+Image_Width-1)>>3)
					 + (*(ON_pixel_cur+j+Image_Width+1)>>3) ;

			pre_self = *(ON_pixel_pre+j) ;

			pre_adjacent = (*(ON_pixel_pre+j-1)>>2)
						 + (*(ON_pixel_pre+j+1)>>2)
						 + (*(ON_pixel_pre+j-Image_Width)>>2)
						 + (*(ON_pixel_pre+j+Image_Width)>>2) ;
			
			pre_diagnal  = (*(ON_pixel_pre+j-Image_Width-1)>>3)
						 + (*(ON_pixel_pre+j-Image_Width+1)>>3)
						 + (*(ON_pixel_pre+j+Image_Width-1)>>3)
						 + (*(ON_pixel_pre+j+Image_Width+1)>>3) ;
			
			I_on = (uint16_t )(alpha_on_self)*self     - (1-alpha_on_self)*(pre_self)
							+ (alpha_on_adj )*adjacent - (1-alpha_on_adj)*(pre_adjacent)
							+ (alpha_on_diag)*diagonal - (1-alpha_on_diag)*(pre_diagnal) ;

			// S : 
			S_on  = *(ON_pixel_cur+j) - (*W_on_bias)*I_on ;
			// S_off = *(ON_pixel_cur+j) - (*W_on_bias)*I_off ;
			if(S_on<0) S_on=0 ;
			// if(S_off<0) S_off=0 ;
			
			// superliner cumputation:
			// *(S_Layer_Pixel+j) = (uint8_t)(theta_1*S_on + theta_2*S_off + theta_3*S_on*S_off ) ;
			*(S_Layer_Pixel+j) = (uint8_t)(theta_1*S_on ) ;


			// Ce : need completed spacial information of S_Layer !
		}
	} // now we have completed S_Layer.

}

// S_Layer with only OFF_Channel
void S_Calculate_OFF(LGMD_pControlTypedef* hLGMD)
{
	LGMDstruct_Layer* Layers = &hLGMD->Model->Layers ;
	LGMDstruct_Params* Params = &hLGMD->Model->Params ;
	LGMDstruct_Result* Result = &hLGMD->Model->Results ;
	uint8_t *S_Layer_Pixel , *G_Layer_Pixel ;

	int8_t* P_Layer_Pixel ;	// point to current P_Layer's pixel
	uint8_t *ON_pixel_cur , *ON_pixel_pre , *OFF_pixel_cur , *OFF_pixel_pre ;
	// P pixel
	P_Layer_Pixel = &Diff_Image[hLGMD->currentDiffImage][0][0] ;	// uint 8
	// ON OFF pixel
	// ON_pixel_cur = &hLGMD->Model->Layers.ON[hLGMD->currentDiffImage][0][0] ;
	// ON_pixel_pre = &hLGMD->Model->Layers.ON[!(hLGMD->currentDiffImage)][0][0] ;
	OFF_pixel_cur = &hLGMD->Model->Layers.OFF[hLGMD->currentDiffImage][0][0] ;
	OFF_pixel_pre = &hLGMD->Model->Layers.OFF[!(hLGMD->currentDiffImage)][0][0] ;

	// Delay_params for I convolution : 
	uint8_t i , j ;
	uint16_t self , adjacent , diagonal , pre_self , pre_adjacent , pre_diagnal ;
	float alpha_on_self , alpha_on_adj , alpha_on_diag , alpha_off_self , alpha_off_adj , alpha_off_diag ;	uint16_t I_on , I_off ;
	float S_on , S_off ;

	// alpha_on_self = Params->alpha_on_self ;
	// alpha_on_adj = Params->alpha_on_adj ;
	// alpha_on_diag = Params->alpha_on_diag ;
	alpha_off_self = Params->alpha_off_self ;
	alpha_off_adj = Params->alpha_off_adj ;
	alpha_off_diag = Params->alpha_off_diag ;

	// Wi coefficient , Bias by FFI :
	float Tffi , W_base_off ; 
	float *W_on_bias ,* W_off_bias ;
	float *cur_ffi_on , *cur_ffi_off ;
	
	// cur_ffi_on = &hLGMD->Model->Results.FFI_ON[hLGMD->currentDiffImage] ;
	cur_ffi_off = &hLGMD->Model->Results.FFI_OFF[hLGMD->currentDiffImage] ;

	Tffi = hLGMD->Model->Params.Tffi ;
	// *W_on_bias = &hLGMD->Model->Params.W_on_bias ;
	W_off_bias = &hLGMD->Model->Params.W_off_bias ;
	W_base_off = hLGMD->Model->Params.W_base_off ;

	// *W_on_bias = ((*cur_ffi_on/Tffi )>W_base ) ? (*cur_ffi_on/Tffi ) : (W_base ) ;
	*W_off_bias = ((*cur_ffi_off/Tffi )>W_base_off ) ? (*cur_ffi_off/Tffi ) : (W_base_off ) ;

	// superliner cumputation : 
	float theta_1 , theta_2 , theta_3 ;
	theta_1 = Params->theta_1 ;
	theta_2 = Params->theta_2 ;
	theta_3 = Params->theta_3 ;

	
	for(i = 1 ; i < Image_Height - 1 ; i ++ )
	{
		// core
		// To use ptr calculation in the second_fold_for_loop :
		P_Layer_Pixel = &Diff_Image[hLGMD->currentDiffImage][i][0] ;

		// ON_pixel_cur = &Layers->ON[hLGMD->currentDiffImage][i][0] ;
		// ON_pixel_pre = &Layers->ON[!(hLGMD->currentDiffImage)][i][0] ;

		OFF_pixel_cur = &Layers->OFF[hLGMD->currentDiffImage][i][0] ;
		OFF_pixel_pre = &Layers->OFF[!(hLGMD->currentDiffImage)][i][0] ;

		// I_Layer_Pixel = &hLGMD->Model->Layers.I[i][0] ;
		S_Layer_Pixel = &hLGMD->Model->Layers.S[i][0] ;
		G_Layer_Pixel = &hLGMD->Model->Layers.G[i][0] ;		

		for(j = 1 ; j < Image_Width - 1 ; j ++ )
		{
			// I_OFF : convolving delayed excitation in OFF channels to form OFF inhibition
			self = *(OFF_pixel_cur+j) ;

			adjacent = (*(OFF_pixel_cur+j-1)>>2)
					 + (*(OFF_pixel_cur+j+1)>>2)
					 + (*(OFF_pixel_cur+j-Image_Width)>>2)
					 + (*(OFF_pixel_cur+j+Image_Width)>>2) ;
			
			diagonal = (*(OFF_pixel_cur+j-Image_Width-1)>>3)
					 + (*(OFF_pixel_cur+j-Image_Width+1)>>3)
					 + (*(OFF_pixel_cur+j+Image_Width-1)>>3)
					 + (*(OFF_pixel_cur+j+Image_Width+1)>>3) ;

			pre_self = *(OFF_pixel_pre+j) ;

			pre_adjacent = (*(OFF_pixel_pre+j-1)>>2)
						 + (*(OFF_pixel_pre+j+1)>>2)
						 + (*(OFF_pixel_pre+j-Image_Width)>>2)
						 + (*(OFF_pixel_pre+j+Image_Width)>>2) ;
			
			pre_diagnal  = (*(OFF_pixel_pre+j-Image_Width-1)>>3)
						 + (*(OFF_pixel_pre+j-Image_Width+1)>>3)
						 + (*(OFF_pixel_pre+j+Image_Width-1)>>3)
						 + (*(OFF_pixel_pre+j+Image_Width+1)>>3) ;
			
			I_off = (uint16_t )(alpha_off_self)*self    - (1-alpha_off_self)*(pre_self)
							+ (alpha_off_adj )*adjacent - (1-alpha_off_adj)*(pre_adjacent)
							+ (alpha_off_diag)*diagonal - (1-alpha_off_diag)*(pre_diagnal) ;

			// S : 
			// S_on  = *(ON_pixel_cur+j) - (*W_on_bias)*I_on ;
			S_off = *(ON_pixel_cur+j) - (*W_on_bias)*I_off ;
			// if(S_on<0) S_on=0 ;
			if(S_off<0) S_off=0 ;
			
			// superliner cumputation:
			// *(S_Layer_Pixel+j) = (uint8_t)(theta_1*S_on + theta_2*S_off + theta_3*S_on*S_off ) ;
			*(S_Layer_Pixel+j) = (uint8_t)(theta_2*S_off ) ;


			// Ce : need completed space information of S_Layer !
		}
	} // now we have completed S_Layer.
}

// S_Layer with no ON_OFF
void S_Calculate_none(LGMD_pControlTypedef* hLGMD) 
{
	// both ON and OFF channel closed: (S_on = 0)&&(S_off = 0) ==> S_Layer = 0
	int8_t* S_Layer_Pixel ;
	for(int i = 0 ; i < Image_Height ; i ++ )
	{
		S_Layer_Pixel = &hLGMD->Model->Layers.S[i][0] ;
		for(int j = 0 ; j < Image_Width ; j ++ )
		{
			*(S_Layer_Pixel+j) = 0 ;
		}
	}
}


uint8_t LGMD_demo(LGMD_pControlTypedef* hLGMD)
{
	if( hLGMD->Enable == 0 )
		return 0 ;

	while( *(hLGMD->hFrameCount) == hLGMD->processCountLast )
		hLGMD->status = 0 ;

	hLGMD->status = 1 ;

	TICin2; //start clocking,reset timer to 0
	hCoS->hHBIO->Instance->timerlog[0]=TOCin2; //save timer tick
	LGMD_Calculating(hLGMD) ;
	hCoS->hHBIO->Instance->timerlog[1]=TOCin2; //save timer tick

	// whole process over , record
	hLGMD->processCountLast = *(hLGMD ->hFrameCount);
	(hLGMD->processCount)++;


	return 0 ; 
}

uint8_t Decision_making(LGMD_pControlTypedef* hLGMD , uint8_t allow_motion)
{
	uint8_t command = 0 ;
	uint8_t Collision = hLGMD->Model->Results.Collision ;
	uint8_t frame = (*hLGMD->hFrameCount)%3 ;
	
	if ( Collision )	//Collision recognition
	{
		LED2_Toggle;
		
		if( frame == 0 )
		{
			command = 'R' ;		
		}
		else
		{
			command = 'L' ;		
		}
	}

	if(allow_motion)
	{
		/************* Pay Attention! ******************/
		//if motion sequence is not empty, do not push motions
		uint8_t readMotionQueueStatus[10]={0};
		ReadBytesWithACKFromColiasBasic(hCoS->hHCos_Motion->Instance->hHUART,0x97,readMotionQueueStatus,1);
		
		// &0x0f :
		// ����λ��������00001111���а�λ�������
		// �������ͨ����������һ�����ĸ���λ����������λ
		if((readMotionQueueStatus[1]&0x0f)==0x09)
		{
			fast_motion_control(hCoS->hHCos_Motion->Instance,command,0);	
		}
	}
	

	return 0;
}



// Founctions used in LGMD_Pro listed below : 
