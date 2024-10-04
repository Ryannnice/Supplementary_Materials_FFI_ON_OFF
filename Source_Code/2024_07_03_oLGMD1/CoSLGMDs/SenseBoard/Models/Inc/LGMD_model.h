/*
 * Description: LGMD-Basic
 * Filename: LGMD_model.h
 * Author: Renyvan Liu
 * Date: 2024 May  
*/ 

#ifndef __LGMD_PC_H 
#define __LGMD_PC_H


#include <stdint.h> 


#define Layer_Height 72
#define Layer_Width 99

typedef struct
{
	uint8_t ON[2][Layer_Height][Layer_Width] ;
	uint8_t OFF[2][Layer_Height][Layer_Width] ;
	// int8_t I[Layer_Height][Layer_Width] ;	//Inhibitory Layer
	int8_t S[Layer_Height][Layer_Width] ;	//Summation Layer
	// uint8_t Ce[Layer_Height][Layer_Width] ;	//Ce for G calculating 
	uint8_t G[Layer_Height][Layer_Width] ;	//Grouping Layer

}LGMDstruct_Layer; // level 3


typedef struct
{
	// ON_OFF :
	float tau_in ; // in: interval between two adjecent frames.
	float clip;	//clip point in ON and OFF mechanisms  // clip=0 , to judge wether goto ON or OFF channel 
	
	float alpha_left_on ;
	float alpha_left_off ; // coefficient of lefted in previous frame.

	float tau_on_self_lowpass ;
	float tau_on_adj_lowpass ;
	float tau_on_diag_lowpass ;
	float alpha_on_self ;
	float alpha_on_adj ;
	float alpha_on_diag ; // Inhibitory for ON

	float tau_off_self_lowpass ;
	float tau_off_adj_lowpass ;
	float tau_off_diag_lowpass ;
	float alpha_off_self ;
	float alpha_off_adj ;
	float alpha_off_diag ; // Excitation for OFF


	// calculating FFI :
	float Tffi ; // FFI Threshould--------> *Boff=*cur_ffi/Tffi; *Bon=*cur_ffi/Tffi;  (B:Bias) 
	// float tau_ffi_lowpass ; // low_pass tau for alpha_FFI 
	// float alpha_ffi ; // delay coefficient in FFI

	/****************************************************************************************************/
	float tau_ffi_on ;
	float tau_ffi_off ;
	
	float alpha_ffi_on ;
	float alpha_ffi_off ;

	float TH_on ;
	float TH_off ;
	/****************************************************************************************************/

	// calculating S :
	float W_base_on ;	// Inhibition coefficient, baseline of local bias  -------->   basic 'W' in "S = E - 'W'*I"
	float W_base_off ;
	float W_on_bias ; 
	float W_off_bias ; // calculated by FFI

	float theta_1 ;
	float theta_2 ;
	float theta_3 ; // for superliner cumputation of "S".

	// G :
	float Delta ; // to prevent "devide 0" case
	float Cw ; // Grouping decaying strength used in "G" calculating
	// G~ :
	float Cde ; // Grouping coefficient
	float Tde ; // Grouping layer Threshould
	// Sigmoid MP (k) :
	// Sigmoid : 
	float Csig ; 
	// My Sigmoid : 
	float Csig1 ;
	float Csig2 ; 
	// SFA : 
	float tau_sfa_highpass ; // high_pass tau for alpha_SFA
	float alpha_sfa ; // delay coefficient in SFA 
	float Tsfa;	//threshold for SFA

	// calculating Spike :
	float Kspi ;//scale parameter in spiking mechanism  ----> floor(exp( Kspi *(*cur_lgmd_out-Tsp ) ) ) 
	float Tspi ;	// threshold for spiking
	float TspiF ; // threshold for spiking frequency inidicating danger  -------> if (spiF>=TspiF): Collision
	// float Nsp ; // Threshould for constant_number of Spike (4)


}LGMDstruct_Params; // level 3


typedef struct
{
	// float FFI[2] ;// store current FFI and previous FFI,while using previous one
	/******************************/
	float FFI_ON[2] ;
	float FFI_OFF[2] ;
	/******************************/
	float MP ; // Kf , sum_G~
	float SMP[2] ; // kf , sigmoid "Kf"
	float LGMD_output[2] ; // SMP after SFA_mechanism 

	// uint8_t Spike ; // 0 / 1
	// uint8_t Constant_spike ; // constant spike number
	// uint8_t Cfinal ; // 0 / 1 
	// float sum_ffi ;
	// uint8_t Spike[10] ; 
	// float Spike_Frequency ; // *spiF=(*spiF)*100/(1000/hLGMD->processRate) ;

	uint8_t Spike ;
	uint8_t constant ;
	uint8_t Collision ; // 0 / 1  ***FINAL RESULT***

	// watch : 
	float cur_ave_ffi_ON ;
	float cur_ave_ffi_OFF ;



}LGMDstruct_Result; // level 3


typedef struct
{
	LGMDstruct_Layer Layers; 
	LGMDstruct_Params Params;
	LGMDstruct_Result Results;
	int16_t* pImg;
	int8_t* pDiffImg;
	
}LGMDType; // level 2

/*
typedef struct
{
	uint8_t ON[2][Layer_Height][Layer_Width];	//ON Cells
	uint8_t OFF[2][Layer_Height][Layer_Width];	//OFF Cells
	uint8_t S[Layer_Height][Layer_Width];	//Summation Layer
	uint8_t G[Layer_Height][Layer_Width];	//Grouping Layer
}LGMDstruct_Layer; //// 3

typedef struct
{
	uint8_t Kspi;	//scale parameter in spiking mechanism
	uint8_t Cw;	//constant to compute the scale in G layer
	uint8_t CdeTde;	//threshold gate in G layer
	uint8_t Tffi;	//FFI threshold
	uint8_t Tspi;	//threshold for spiking frequency indicating danger
	int8_t clip;	//clip point in ON and OFF mechanisms
	float W_on;	//weight on ON channel excitation
	float W_off;	//weight on OFF channel excitation
	float W_base;	//baseline of local bias
	float Tsp;	//threshold for spiking
	float Tsfa;	//threshold for SFA
	float Bias_on;	//inhibitory bias in ON channels
	float Bias_off;	//inhibitory bias in OFF channels
	float Ksig;	//coefficient in sigmoid function
	float DeltaC;	//small real number to compute the scale in G layer
	float hp_tau;	//latency in SFA
	float lp_ffi;	//latency in FFI-M
	float lp_on_cent_tau;	//latency in centre cells in ON channels
	float lp_on_near_tau;	//latency in nearest cells in ON channels
	float lp_on_diag_tau;	//latency in diagonal cells in ON channels
	float lp_off_cent_tau;	//latency in centre cells in OFF channels
	float lp_off_near_tau;	//latency in nearest cells in OFF channels
	float lp_off_diag_tau;	//latency in diagonal cells in OFF channels
	float delay_sfa;	//delay coefficient in SFA
	float delay_ffi;	//delay coefficient in FFI
	float delay_on_cent;	//delay coefficient in centre cells in ON channels
	float delay_on_near;	//delay coefficient in nearest cells in ON channels
	float delay_on_diag;	//delay coefficient in diagonal cells in ON channels
	float delay_off_cent;	//delay coefficient in centre cells in OFF channels
	float delay_off_near;	//delay coefficient in nearest cells in OFF channels
	float delay_off_diag;	//delay coefficient in diagonal cells in OFF channels
}LGMDstruct_Params;


typedef struct
{/////////////////////////////////
	uint8_t SPIKE[10];	//spikes                  ~    ~    ~
	float MP;	//membrane potential  <--> Kf ( = G1 + G2 + G3 + ...)
	float FFI_out[2];	//FFI output
	float SMP[2];	//sigmoid membrane potential  <--> kf ( = )
	float LGMD_out[2];	//LGMDs output
	float spiFrequency;	//spiking frequency
}LGMDstruct_Result;

typedef struct
{
	LGMDstruct_Layer Layers; //// 3
	LGMDstruct_Params Params;
	LGMDstruct_Result Results;
	int16_t* pImg;
	int8_t* pDiffImg;
}LGMDType; //// 2

*/

#endif

