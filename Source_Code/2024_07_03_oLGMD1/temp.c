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
			
			pre_diagnal = (*(ON_pixel_pre+j-Image_Width-1)>>2)
						 + (*(ON_pixel_pre+j-Image_Width+1)>>2)
						 + (*(ON_pixel_pre+j+Image_Width-1)>>2)
						 + (*(ON_pixel_pre+j+Image_Width+1)>>2) ;
			
			I_on = (uint16_t )(alpha_on_self)*self     - (1-alpha_on_self)*(pre_self)
							+ (alpha_on_adj )*adjacent - (1-alpha_on_adj)*(pre_adjacent)
							+ (alpha_on_diag)*diagonal - (1-alpha_on_diag)*(pre_diagnal)
