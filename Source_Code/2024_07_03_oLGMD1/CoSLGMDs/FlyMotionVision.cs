/*
 * Filename: FlyMotionVision.cs
 * Author: Qinbing Fu
 * Location: Guangzhou
 * Date: 2021 Jan & Mar & Apr
 */


using System;


namespace EMD
{
    /// <summary>
    /// This is a base class.
    /// This class implements Drosophila motion vision neural networks including translating perception LPTCs, ultra-selective looming perception LPLC2 neurons.
    /// This class also implements a hybrid network for both foreground object and background motion perception.
    /// This neural network model has novelties on several aspects:
    /// 1.  Instantaneous contrast normalization
    /// 2.  Two forms of triple correlation
    /// 3.  A comprehensive, multi-pathway motion vision neural network model: motion vision LPTC and LPLC2, contrast vision
    /// 4.  Fractional order
    /// 5.  Postsynaptic neural network output
    /// 6.  Negative feedback
    /// </summary>
    internal class FlyMotionVision
    {
        #region NETWORK FIELD
        /// <summary>
        /// Sampling distance in space
        /// </summary>
        protected readonly int sd;
        /// <summary>
        /// Input frame width
        /// </summary>
        protected readonly int width;
        /// <summary>
        /// Input frame height
        /// </summary>
        protected readonly int height;
        /// <summary>
        /// Whether blocking ON channels
        /// </summary>
        protected readonly bool blockON;
        /// <summary>
        /// Whether blocking OFF channels
        /// </summary>
        protected readonly bool blockOFF;
        /// <summary>
        /// Whether blocking contrast vision
        /// </summary>
        protected readonly bool blockContrast;
        /// <summary>
        /// Whether blocking LPTC
        /// </summary>
        protected readonly bool blockLPTC;
        /// <summary>
        /// Whether blocking LPLC2
        /// </summary>
        protected readonly bool blockLPLC2;
        /// <summary>
        /// Clip point in half-wave rectifying
        /// </summary>
        protected readonly double clip;
        /// <summary>
        /// DC component in half-wave rectifying
        /// </summary>
        protected readonly double dc;
        /// <summary>
        /// Time constant in high-pass filtering of Retina layer in milliseconds
        /// </summary>
        protected readonly int tau_hp;
        /// <summary>
        /// Time constant in low-pass filtering of signal correlation in milliseconds
        /// </summary>
        protected readonly int tau_lp;
        /// <summary>
        /// Time constant in low-pass filtering of T4 neurons to LPTC in milliseconds
        /// </summary>
        protected readonly int t4_lp;
        /// <summary>
        /// Time constant in low-pass filtering of T5 neurons to LPTC in milliseconds
        /// </summary>
        protected readonly int t5_lp;
        /// <summary>
        /// Delay coefficient in Retina layer
        /// </summary>
        protected readonly double delay_hp;
        /// <summary>
        /// Delay coefficient in Medulla layer
        /// </summary>
        protected readonly double delay_lp;
        /// <summary>
        /// Delay coefficient in T4 neurons
        /// </summary>
        protected readonly double delay_t4;
        /// <summary>
        /// Delay coefficient in T5 neurons
        /// </summary>
        protected readonly double delay_t5;



        /// <summary>
        /// Exponent of T4 neuron response
        /// </summary>
        protected readonly double exp_ON;
        /// <summary>
        /// Exponent of T5 neuron response
        /// </summary>
        protected readonly double exp_OFF;
        /// <summary>
        /// Standard deviation in excitatory field
        /// </summary>
        protected readonly double std_exc;
        /// <summary>
        /// Standard deviation in inhibitory field
        /// </summary>
        protected readonly double std_inh;
        /// <summary>
        /// Standard deviation in contrast normalisation field
        /// </summary>
        protected readonly double std_cn;
        /// <summary>
        /// Coefficient in contrast normalisation field, = 10
        /// </summary>
        protected readonly double coe_cn;
        /// <summary>
        /// Width of convolution kernel in contrast normalisation field
        /// </summary>
        protected readonly int Wcn;
        /// <summary>
        /// Width of convolution kernel in excitation field
        /// </summary>
        protected readonly int Wexc;
        /// <summary>
        /// Width of convolution kernel in inhibition field
        /// </summary>
        protected readonly int Winh;

        
        /// <summary>
        /// Bias in two-arm motion signal correlation
        /// </summary>
        protected readonly double Mbias;
        /// <summary>
        /// Interval between discrete frames
        /// </summary>
        protected readonly double interval;
        /// <summary>
        /// Coefficient of parallel motion pathway in summation, = 1
        /// </summary>
        protected readonly double coe_motion;
        /// <summary>
        /// Coefficient of parallel contrast pathway in summation, = 1
        /// </summary>
        protected readonly double coe_contrast;
        /// <summary>
        /// Standard deviation (horizontal) in radial spatial bias distribution
        /// </summary>
        protected readonly double std_w;
        /// <summary>
        /// Standard deviation (vertical) in radial spatial bias distribution
        /// </summary>
        protected readonly double std_h;
        /// <summary>
        /// Scale parameter (horizontal) in making Gaussian distribution of the radial spatial bias
        /// </summary>
        protected readonly double scale_w;
        /// <summary>
        /// Scale parameter (vertical) in making Gaussian distribution of the radial spatial bias
        /// </summary>
        protected readonly double scale_h;
        /// <summary>
        /// Time window in frames for averaging
        /// </summary>
        protected readonly int time_length;
        #endregion

        #region NETWORK LAYER
        /// <summary>
        /// Photoreceptor(Retina) layer - Dimension width * height * 2
        /// </summary>
        protected double[,,] Photoreceptor;
        /// <summary>
        /// Gaussian blur layer - Dimension width * height
        /// </summary>
        protected double[,] GB;
        /// <summary>
        /// Gaussian blur convolution kernel in excitation field - Dimension Wexc * Wexc
        /// </summary>
        protected double[,] GB_ExcKernel;
        /// <summary>
        /// Gaussian blur convolution kernel in inhibition field - Dimension Winh * Winh
        /// </summary>
        protected double[,] GB_InhKernel;
        /// <summary>
        /// ON channels - Dimension width * height * 2
        /// </summary>
        protected double[,,] ONs;
        /// <summary>
        /// OFF channels - Dimension width * height * 2
        /// </summary>
        protected double[,,] OFFs;
        /// <summary>
        /// ON channels compressed signal - Dimension width * height * 2
        /// </summary>
        protected double[,,] ONs_Compressed;
        /// <summary>
        /// OFF channels compressed signal - Dimension width * height * 2
        /// </summary>
        protected double[,,] OFFs_Compressed;
        /// <summary>
        /// Convolution kernel in contrast normalization
        /// </summary>
        protected double[,] Kernel_Constrast;
        /// <summary>
        /// Contrast vision in ON channels - Dimension width * height * 2
        /// </summary>
        protected double[,,] ONs_Contrast;
        /// <summary>
        /// Contrast vision in OFF channels - Dimension width * height * 2
        /// </summary>
        protected double[,,] OFFs_Contrast;
        /// <summary>
        /// ON channels delayed signal - Dimension width * height * 2
        /// </summary>
        protected double[,,] ONs_delay;
        /// <summary>
        /// OFF channels delayed signal - Dimension width * height * 2
        /// </summary>
        protected double[,,] OFFs_delay;
        /// <summary>
        /// T4 neurons in Medulla - Dimension width * height * 2 * 4 (cardinal directions), 0->rightward, 1->leftward, 2->downward, 3->upward
        /// </summary>
        protected double[,,,] T4s;
        /// <summary>
        /// T5 neurons in Lobula - Dimension width * height * 2 * 4 (cardinal directions), 0->rightward, 1->leftward, 2->downward, 3->upward
        /// </summary>
        protected double[,,,] T5s;
        /// <summary>
        /// LPTCs in Lobula Plate - Dimension 4 * 1 vector, 0->rightward, 1->leftward, 2->downward, 3->upward
        /// </summary>
        protected double[] LPTCs;
        /// <summary>
        /// Lobula Plate intrinsic interneurons inhibiting stratified LPTCs - Dimension 4 * 1 vector, 0->leftward, 1->rightward, 2->upward, 3->downward
        /// </summary>
        protected double[] LPis;
        /// <summary>
        /// System of horiztonal sensitive
        /// </summary>
        protected double SysHS;
        /// <summary>
        /// System of vertical sensitive
        /// </summary>
        protected double SysVS;
        /// <summary>
        /// Visual Projection Neurons LPLC2 response
        /// </summary>
        protected double[] LPLC2;
        /// <summary>
        /// Averaged output response of LPLC2 visual projection neurons
        /// </summary>
        protected double LPLC2_Out;
        /// <summary>
        /// Motion Signal Map (RGB) of T4 neurons
        /// </summary>
        protected byte[,,] T4_msm;
        /// <summary>
        /// Motion Signal Map (RGB) of T5 neurons
        /// </summary>
        protected byte[,,] T5_msm;
        /// <summary>
        /// Max motion of T4 neurons
        /// </summary>
        protected double T4_maxMag;
        /// <summary>
        /// Max motion of T5 neurons
        /// </summary>
        protected double T5_maxMag;
        /// <summary>
        /// Direction indicator matrix of T4 neurons
        /// </summary>
        protected double[,] T4_dir;
        /// <summary>
        /// Direction indicator matrix of T5 neurons
        /// </summary>
        protected double[,] T5_dir;
        /// <summary>
        /// Motion magnitude matrix of T4 neurons
        /// </summary>
        protected double[,] T4_mag;
        /// <summary>
        /// Motion magnitude matrix of T5 neurons
        /// </summary>
        protected double[,] T5_mag;
        /// <summary>
        /// LPTCs motion magnitude
        /// </summary>
        protected double[,] LPTCs_mag;
        /// <summary>
        /// LPTCs motion direction
        /// </summary>
        protected double[,] LPTCs_dir;
        /// <summary>
        /// LPTCs motion signal map
        /// </summary>
        protected byte[,,] LPTCs_msm;
        /// <summary>
        /// Regional response of whole population of LPLC2 visual projection neurons - Dimension 4 * 1 vector, 0->left_up, 1->right_up, 2->left_bottom, 3->right_bottom
        /// </summary>
        protected double[] LPLC2_Region;
        /// <summary>
        /// Centroid of view - Dimension 2 * 1 vector, 0->abscissa, 1->ordinate
        /// </summary>
        protected int[] View_Centroid;
        /// <summary>
        /// Spatial bias distribution on radial outward motion
        /// </summary>
        protected double[,] Radial_Bias;
        /// <summary>
        /// LPLC2 motion signal map
        /// </summary>
        protected byte[,,] LPLC2_msm;
        /// <summary>
        /// LPLC2 motion magnitude
        /// </summary>
        protected double[,] LPLC2_mag;
        /// <summary>
        /// LPLC2 motion direction
        /// </summary>
        protected double[,] LPLC2_dir;
        #endregion

        #region NETWORK CONSTRUCTOR
        /// <summary>
        /// Constructor
        /// </summary>
        public FlyMotionVision()
        { }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="width"></param>
        /// <param name="height"></param>
        /// <param name="fps"></param>
        public FlyMotionVision(int width /*frame width*/, int height /*frame height*/, int fps /*frames per second*/)
        {
            this.width = width;
            this.height = height;
            interval = 1000 / fps;
            sd = 1;
            blockON = false;
            blockOFF = false;
            blockLPTC = false;
            blockContrast = false;
            blockLPLC2 = false;
            clip = 0.1;
            dc = 0.1;
            tau_hp = 500;
            tau_lp = 30;
            t4_lp = 30;
            t5_lp = 30;
            delay_hp = tau_hp / (tau_hp + interval);
            delay_lp = interval / (tau_lp + interval);
            delay_t4 = interval / (t4_lp + interval);
            delay_t5 = interval / (t5_lp + interval);
            exp_ON = 0.9;
            exp_OFF = 0.5;
            std_exc = 1;
            std_inh = 2;
            std_cn = 5;
            coe_cn = 10; //10 d, 20 dd
            coe_contrast = 1;
            coe_motion = 1;
            Wcn = 11;
            Wexc = 3;
            Winh = 5;
            Mbias = 0.89;
            std_w = 1;
            std_h = std_w * height / width;
            scale_w = 3 * std_w;
            scale_h = 3 * std_h;
            time_length = 10;
            //
            Photoreceptor = new double[this.height, this.width, 2];
            GB = new double[this.height, this.width];
            GB_ExcKernel = new double[Wexc, Wexc];
            GB_InhKernel = new double[Winh, Winh];
            ONs = new double[this.height, this.width, 2];
            OFFs = new double[this.height, this.width, 2];
            ONs_Compressed = new double[this.height, this.width, 2];
            OFFs_Compressed = new double[this.height, this.width, 2];
            ONs_Contrast = new double[this.height, this.width, 2];
            OFFs_Contrast = new double[this.height, this.width, 2];
            ONs_delay = new double[this.height, this.width, 2];
            OFFs_delay = new double[this.height, this.width, 2];
            Kernel_Constrast = new double[Wcn, Wcn];
            T4s = new double[this.height, this.width, 2, 4];
            T5s = new double[this.height, this.width, 2, 4];
            LPTCs = new double[4];
            LPis = new double[4];
            T4_msm = new byte[this.height, this.width, 3];
            T5_msm = new byte[this.height, this.width, 3];
            T4_mag = new double[this.height, this.width];
            T5_mag = new double[this.height, this.width];
            T4_dir = new double[this.height, this.width];
            T5_dir = new double[this.height, this.width];
            LPLC2_Region = new double[4];
            LPLC2 = new double[time_length];
            View_Centroid = new int[2];
            View_Centroid[0] = height / 2;
            View_Centroid[1] = width / 2;
            Radial_Bias = new double[this.height, this.width];
            LPLC2_msm = new byte[height, width, 3];
            LPLC2_mag = new double[height, width];
            LPLC2_dir = new double[height, width];
            LPTCs_msm = new byte[height, width, 3];
            LPTCs_mag = new double[height, width];
            LPTCs_dir = new double[height, width];
            //
            MakeGaussianKernel(std_exc, Wexc, ref GB_ExcKernel);
            MakeGaussianKernel(std_inh, Winh, ref GB_InhKernel);
            MakeGaussianKernel(std_cn, Wcn, ref Kernel_Constrast);
            MakeRadialBiasDistribution(std_h, std_w, width, height, scale_w, scale_h, ref Radial_Bias);

            Console.WriteLine("Fly moiton vision neural network model constructed......");
        }
        #endregion

        #region NETWORK PROPERTY
        public double[,] T4_MAG
        {
            get { return T4_mag; }
        }
        public double[,] T5_MAG
        {
            get { return T5_mag; }
        }
        public byte[,,] T4_MSM
        {
            get { return T4_msm; }
        }
        public byte[,,] T5_MSM
        {
            get { return T5_msm; }
        }
        public double[] LPTC_OUT
        {
            get { return LPTCs; }
        }
        public double LPLC2_OUT
        {
            get { return LPLC2_Out; }
        }
        public double HS
        {
            get { return SysHS; }
        }
        public double VS
        {
            get { return SysVS; }
        }
        public double[] LPLC2_REGION
        {
            get { return LPLC2_Region; }
        }
        public byte[,,] LPLC2_MSM
        {
            get { return LPLC2_msm; }
        }
        public byte[,,] LPTCS_MSM
        {
            get { return LPTCs_msm; }
        }
        #endregion

        #region ACTIVATION FUNCTION
        /// <summary>
        /// Rectified Linear Unit (ReLU)
        /// </summary>
        /// <param name="input">input node value</param>
        /// <returns></returns>
        protected double ReLU(double input)
        {
            if (input < 0)
                return 0;
            else
                return input;
        }
        /// <summary>
        /// Leaky Rectified Linear Unit (Leaky-ReLU)
        /// </summary>
        /// <param name="input">input node value</param>
        /// <returns></returns>
        protected double Leaky_ReLU(double input)
        {
            if (input >= 0)
                return input;
            else
                return 0.01 * input;
        }
        /// <summary>
        /// Sigmoid Activation
        /// </summary>
        /// <param name="input">input node value</param>
        /// <param name="a">scale coefficient</param>
        /// <returns></returns>
        protected double Sigmoid(double input, double a)
        {
            return Math.Pow(1 + Math.Exp(-input * Math.Pow(a, -1)), -1);
        }
        /// <summary>
        /// Signum function
        /// </summary>
        /// <param name="input">input node value</param>
        /// <returns></returns>
        protected int Signum(double input)
        {
            if (input > 0)
                return 1;
            else if (input == 0)
                return 0;
            else
                return -1;
        }
        /// <summary>
        /// Hyberbolic Tangent function
        /// </summary>
        /// <param name="input">input node value</param>
        /// <returns></returns>
        protected double TanH(double input)
        {
            return (Math.Exp(2 * input) - 1) / (Math.Exp(2 * input) + 1);
        }
        /// <summary>
        /// Gaussian Error Linear Units (GELUs) in slower but more precise approximation
        /// </summary>
        /// <param name="input">input node value</param>
        /// <returns></returns>
        protected double GELU_Precise(double input)
        {
            return 0.5 * input * (1 + TanH(input * 0.7978845608 * (1 + 0.044715 * input * input)));
        }
        /// <summary>
        /// Gaussian Error Linear Units (GELUs) in fast approximation
        /// </summary>
        /// <param name="input">input node value</param>
        /// <returns></returns>
        protected double GELU_Fast(double input)
        {
            return Sigmoid(1.702 * input, 1) * input;
        }
        /// <summary>
        /// Softmax function in neural network classifier
        /// </summary>
        /// <param name="inVector">input vector</param>
        /// <returns></returns>
        protected double[] Softmax(double[] inVector)
        {
            double[] prob = new double[inVector.Length];
            double sumExp = 0;
            for (int i = 0; i < inVector.Length; i++)
            {
                sumExp += Math.Exp(inVector[i]);
            }
            int index = 0;
            foreach (double i in inVector)
            {
                prob[index] = Math.Exp(i) / sumExp;
                index++;
            }
            return prob;
        }
        #endregion

        #region METHOD
        /// <summary>
        /// Make Gaussian Convolution Kernel (mean as 0)
        /// </summary>
        /// <param name="sigma">standard deviation,</param>
        /// <param name="gauss_width">gaussian kernel width</param>
        /// <param name="gauss_kernel">gaussian kernel matrix</param>
        protected void MakeGaussianKernel(double sigma, int gauss_width, ref double[,] gauss_kernel)
        {
            //coordinate of center point
            int center_x = gauss_width / 2;
            int center_y = gauss_width / 2;
            double gaussian; //record different gaussian value according to different sigma
            int distance;
            double sum = 0;
            for (int i = 0; i < gauss_width; i++)
            {
                for (int j = 0; j < gauss_width; j++)
                {
                    distance = (center_x - i) * (center_x - i) + (center_y - j) * (center_y - j);
                    gaussian = Math.Exp((0 - distance) / (2 * sigma * sigma)) / (2 * Math.PI * sigma * sigma);
                    sum += gaussian;
                    gauss_kernel[i, j] = gaussian;
                }
            }
            for (int i = 0; i < gauss_width; i++)
            {
                for (int j = 0; j < gauss_width; j++)
                {
                    gauss_kernel[i, j] /= sum;
                }
            }
        }
        /// <summary>
        /// Make radial spatial bias distribution
        /// </summary>
        /// <param name="sigma1">standard deviation in density function</param>
        /// <param name="sigma2">standard deviation in density function</param>
        /// <param name="gauss_width">width of gaussian</param>
        /// <param name="gauss_height">height of gaussian</param>
        /// <param name="scale_w">scale on width from image to density</param>
        /// <param name="scale_h">scale on height from image to density</param>
        /// <param name="gauss_kernel">kernel matrix</param>
        protected void MakeRadialBiasDistribution(double sigma1, double sigma2, int gauss_width, int gauss_height, double scale_w, double scale_h, ref double[,] gauss_kernel)
        {
            //centriod
            int centroidX = gauss_height / 2;
            int centroidY = gauss_width / 2;
            double gaussian;
            //float distance;
            double distance1;
            double distance2;
            double scale_x;
            double scale_y;
            for (int i = 0; i < gauss_height; i++)
            {
                for (int j = 0; j < gauss_width; j++)
                {
                    //attention
                    //scale x and y within the defined mean-sigma related range of gaussian distribution density function
                    scale_x = (i - centroidX) * scale_h / centroidX;
                    scale_y = (j - centroidY) * scale_w / centroidY;
                    //distance = scale_x * scale_x + scale_y * scale_y;
                    distance1 = scale_x * scale_x;
                    distance2 = scale_y * scale_y;
                    gaussian = Math.Exp(-0.5 * ((distance1 / (sigma1 * sigma1)) + (distance2 / (sigma2 * sigma2)))) / (2 * Math.PI * sigma1 * sigma2);
                    gauss_kernel[i, j] = gaussian;
                }
            }
        }
        /// <summary>
        ///  Gaussian Blur and Subtraction (return double-number matrix)
        /// </summary>
        /// <param name="input">input signal matrix</param>
        /// <param name="gauss_width1">width of gaussian kernel 1</param>
        /// <param name="gauss_kernel1">gaussian kernel 1</param>
        /// <param name="gauss_width2">width of gaussian kernel 2</param>
        /// <param name="gauss_kernel2">gaussian kernel 2</param>
        /// <param name="t">timestamp</param>
        /// <returns></returns>
        protected double[,] Gaussian_Filter(double[,,] input, int gauss_width1, double[,] gauss_kernel1, int gauss_width2, double[,] gauss_kernel2, int t)
        {
            double[,] output1 = new double[height, width];
            double[,] output2 = new double[height, width];
            double[,] result = new double[height, width];
            double tmp;
            //kernel radius
            int k_radius1 = gauss_width1 / 2;
            int k_radius2 = gauss_width2 / 2;
            int r, c;
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    // first blur
                    for (int i = -k_radius1; i < k_radius1 + 1; i++)
                    {
                        for (int j = -k_radius1; j < k_radius1 + 1; j++)
                        {
                            r = y + i;
                            c = x + j;
                            //if exceeding range, let it equal to near pixel
                            while (r < 0)
                            { r++; }
                            while (r >= height)
                            { r--; }
                            while (c < 0)
                            { c++; }
                            while (c >= width)
                            { c--; }
                            //************************
                            tmp = input[r, c, t] * gauss_kernel1[i + k_radius1, j + k_radius1];
                            output1[y, x] += tmp;
                        }
                    }
                    // second blur
                    for (int i = -k_radius2; i < k_radius2 + 1; i++)
                    {
                        for (int j = -k_radius2; j < k_radius2 + 1; j++)
                        {
                            r = y + i;
                            c = x + j;
                            //if exceeding range, let it equal to near pixel
                            while (r < 0)
                            { r++; }
                            while (r >= height)
                            { r--; }
                            while (c < 0)
                            { c++; }
                            while (c >= width)
                            { c--; }
                            //************************
                            tmp = input[r, c, t] * gauss_kernel2[i + k_radius2, j + k_radius2];
                            output2[y, x] += tmp;
                        }
                    }
                    // subtraction with polarity selectivity
                    result[y, x] = output1[y, x] - output2[y, x]; // S = E - I

                    if (output1[y, x] >= 0 && output2[y, x] >= 0)
                    {
                        result[y, x] = Math.Abs(result[y, x]);
                    }
                    else if (output1[y, x] <= 0 && output2[y, x] <= 0)
                    {
                        if (result[y, x] > 0)
                            result[y, x] = -result[y, x];
                    }
                }
            }
            return result;
        }
        /// <summary>
        /// Contrast normalization
        /// </summary>
        /// <param name="kernel">convolution kernel</param>
        /// <param name="inSignal">input signal matrix</param>
        /// <param name="x">abscissa</param>
        /// <param name="y">ordinate</param>
        /// <param name="t">time</param>
        /// <returns></returns>
        protected double normaliseContrast(double[,] kernel, double[,,] inSignal, int x, int y, int t)
        {
            double tmp = 0;
            int np = kernel.GetLength(0) / 2;
            int r, c;
            for (int i = -np; i < np + 1; i++)
            {
                r = x + i;
                while (r < 0)
                    r += 1;
                while (r >= height)
                    r -= 1;
                for (int j = -np; j < np + 1; j++)
                {
                    c = y + j;
                    while (c < 0)
                        c += 1;
                    while (c >= width)
                        c -= 1;
                    tmp += inSignal[r, c, t] * kernel[i + np, j + np];
                }
            }
            // Activation
            return TanH(inSignal[x, y, t] / (coe_cn + tmp));
        }
        /// <summary>
        /// Diverging triple correlation
        /// </summary>
        /// <param name="cur_p">current position signal</param>
        /// <param name="cur_p_d">currrent position+distance signal</param>
        /// <param name="pre_p">previous position signal</param>
        /// <param name="pre_p_d">previous position+distance signal</param>
        /// <returns></returns>
        protected double Div_TripleCorrelation(double cur_p, double cur_p_d, double pre_p, double pre_p_d)
        {
            return pre_p * cur_p * cur_p_d - cur_p * pre_p_d * cur_p_d;
        }
        /// <summary>
        /// Diverging triple correlation with bias on opposing motion
        /// </summary>
        /// <param name="cur_p">current position signal</param>
        /// <param name="cur_p_d">currrent position+distance signal</param>
        /// <param name="pre_p">previous position signal</param>
        /// <param name="pre_p_d">previous position+distance signal</param>
        /// <param name="bias">local bias</param>
        /// <returns></returns>
        protected double Div_TripleCorrelation_Bias(double cur_p, double cur_p_d, double pre_p, double pre_p_d, double bias)
        {
            return bias * (pre_p * cur_p * cur_p_d - cur_p * pre_p_d * cur_p_d);
        }
        /// <summary>
        /// Converging triple correlation
        /// </summary>
        /// <param name="cur_p">current position signal</param>
        /// <param name="cur_p_d">current position+distance signal</param>
        /// <param name="pre_p">previous position signal</param>
        /// <param name="pre_p_d">previous position+distance signal</param>
        /// <returns></returns>
        protected double Cov_TripleCorrelation(double cur_p, double cur_p_d, double pre_p, double pre_p_d)
        {
            return pre_p * pre_p_d * cur_p_d - pre_p * cur_p * pre_p_d;
        }
        /// <summary>
        /// Converging triple correlation with bias on opposing motion
        /// </summary>
        /// <param name="cur_p">current position signal</param>
        /// <param name="cur_p_d">current position+distance signal</param>
        /// <param name="pre_p">previous position signal</param>
        /// <param name="pre_p_d">previous position+distance signal</param>
        /// <param name="bias">local bias</param>
        /// <returns></returns>
        protected double Cov_TripleCorrelation_Bias(double cur_p, double cur_p_d, double pre_p, double pre_p_d, double bias)
        {
            return bias * (pre_p * pre_p_d * cur_p_d - pre_p * cur_p * pre_p_d);
        }
        /// <summary>
        /// HR correlation
        /// </summary>
        /// <param name="cur_p">current position signal</param>
        /// <param name="cur_p_d">current position+distance signal</param>
        /// <param name="pre_p">previous position signal</param>
        /// <param name="pre_p_d">previous position+distance signal</param>
        /// <param name="bias"></param>
        /// <returns></returns>
        protected double HRCorrelation(double cur_p, double cur_p_d, double pre_p, double pre_p_d, double bias)
        {
            return pre_p * cur_p_d - bias * cur_p * pre_p_d;
        }
        /// <summary>
        /// Highpass in Retina layer
        /// </summary>
        /// <param name="pre_in">previous input signal</param>
        /// <param name="cur_in">current input signal</param>
        /// <param name="pre_out">previous output signal</param>
        /// <param name="a">delay coefficient</param>
        /// <returns></returns>
        protected double Highpass(double pre_in, double cur_in, double pre_out, double a)
        {
            return a * (pre_out + cur_in - pre_in);
        }
        /// <summary>
        /// Highpass in Contrast vision
        /// </summary>
        /// <param name="pre_in">previous input signal</param>
        /// <param name="cur_in">current input signal</param>
        /// <returns></returns>
        protected double Highpass(double pre_in, double cur_in)
        {
            return cur_in - pre_in;
        }
        /// <summary>
        /// Lowpass for time delay
        /// </summary>
        /// <param name="cur_in">current input signal</param>
        /// <param name="pre_out">previous output signal</param>
        /// <param name="a">delay coefficient</param>
        /// <returns></returns>
        protected double Lowpass(double cur_in, double pre_out, double a)
        {
            return a * cur_in + (1 - a) * pre_out;
        }
        /// <summary>
        /// Convolution
        /// </summary>
        /// <param name="x">abscissa</param>
        /// <param name="y">ordinate</param>
        /// <param name="matrix">input signal matrix</param>
        /// <param name="kernel">convolution kernel matrix</param>
        /// <param name="R">radius of convolution kernel</param>
        /// <returns></returns>
        protected double Convolving(int x, int y, double[,] matrix, double[,] kernel, int R)
        {
            double tmp = 0;
            int r, c;
            for (int i = -R; i < R + 1; i++)
            {
                r = x + i;
                while (r < 0)
                    r += 1;
                while (r >= height)
                    r -= 1;
                for (int j = -R; j < R + 1; j++)
                {
                    c = y + j;
                    while (c < 0)
                        c += 1;
                    while (c >= width)
                        c -= 1;
                    tmp += matrix[r, c] * kernel[i + R, j + R];
                }
            }
            return tmp;
        }
        /// <summary>
        /// Convolution with respect to time
        /// </summary>
        /// <param name="x">abscissa</param>
        /// <param name="y">ordinate</param>
        /// <param name="matrix">input signal matrix</param>
        /// <param name="kernel">convolution kernel matrix</param>
        /// <param name="R">radius of convolution kernel</param>
        /// <param name="t">time</param>
        /// <returns></returns>
        protected double Convolving(int x, int y, double[,,] matrix, double[,] kernel, int R, int t)
        {
            double tmp = 0;
            int r, c;
            for (int i = -R; i < R + 1; i++)
            {
                r = x + i;
                while (r < 0)
                    r += 1;
                while (r >= height)
                    r -= 1;
                for (int j = -R; j < R + 1; j++)
                {
                    c = y + j;
                    while (c < 0)
                        c += 1;
                    while (c >= width)
                        c -= 1;
                    tmp += matrix[r, c, t] * kernel[i + R, j + R];
                }
            }
            return tmp;
        }
        /// <summary>
        /// Polarity channels summation each with fractional order
        /// </summary>
        /// <param name="on_exc">ON channel excitation</param>
        /// <param name="off_exc">OFF channel excitation</param>
        /// <param name="on_exp">exponent of ON channel output</param>
        /// <param name="off_exp">exponent of OFF channel output</param>
        /// <returns></returns>
        protected double PolarityChannelSummation(double on_exc, double off_exc, double on_exp, double off_exp)
        {
            return Math.Pow(on_exc, on_exp) + Math.Pow(off_exc, off_exp);
        }
        /// <summary>
        /// Halfwave rectifying in terms of onset response
        /// </summary>
        /// <param name="cur_in">current input signal</param>
        /// <param name="pre_out">previous output signal</param>
        /// <returns></returns>
        protected double Halfwave_ON(double cur_in, double pre_out)
        {
            if (cur_in >= clip)
                return cur_in + dc * pre_out;
            else
                return dc * pre_out;
        }
        /// <summary>
        /// Halfwave rectifying in terms of offset response
        /// </summary>
        /// <param name="cur_in">current input signal</param>
        /// <param name="pre_out">previous output signal</param>
        /// <returns></returns>
        protected double Halfwave_OFF(double cur_in, double pre_out)
        {
            if (cur_in < clip)
                return Math.Abs(cur_in) + dc * pre_out;
            else
                return dc * pre_out;
        }
        /// <summary>
        /// Lipetz transfer
        /// </summary>
        /// <param name="pre_output">previous output signal</param>
        /// <param name="cur_input">current input signal</param>
        /// <param name="delay">delay coefficient</param>
        /// <param name="exp">exponent</param>
        /// <returns></returns>
        protected double LipetzTransfer(double pre_output, double cur_input, double delay, double exp)
        {
            // To compute an adaption state
            if (cur_input == 0)
                return 0;
            else
            {
                double mid = Lowpass(cur_input, pre_output, delay);
                return Math.Pow(cur_input, exp) / (Math.Pow(cur_input, exp) + Math.Pow(mid, exp));
            }
        }
        /// <summary>
        /// Fast Depolarisation Slow Repolarisation Mechanism
        /// </summary>
        /// <param name="pre_on">previous ON cell input</param>
        /// <param name="cur_on">current ON cell input</param>
        /// <param name="pre_off">previous OFF cell input</param>
        /// <param name="cur_off">current OFF cell input</param>
        /// <param name="fdsr_fast_delay">fast delay coefficient in FDSR</param>
        /// <param name="fdsr_slow_delay">slow delay coefficient in FDSR</param>
        protected void FDSR(double pre_on, ref double cur_on, double pre_off, ref double cur_off, double fdsr_fast_delay, double fdsr_slow_delay)
        {
            //calculate the change gradients
            double on_grad = cur_on - pre_on;
            double off_grad = cur_off - pre_off;
            //fast onset slow decay
            if (on_grad >= 0)
                cur_on = cur_on - Lowpass(cur_on, pre_on, fdsr_fast_delay);
            else
                cur_on = cur_on - Lowpass(cur_on, pre_on, fdsr_slow_delay);
            if (off_grad >= 0)
                cur_off = cur_off - Lowpass(cur_off, pre_off, fdsr_fast_delay);
            else
                cur_off = cur_off - Lowpass(cur_off, pre_off, fdsr_slow_delay);
            //positive outputs
            if (cur_on < 0)
                cur_on = 0;
            if (cur_off < 0)
                cur_off = 0;
        }
        /// <summary>
        /// Contrast vision from Medulla layer (Inhibition)
        /// </summary>
        /// <param name="x">abscissa</param>
        /// <param name="y">ordinate</param>
        /// <param name="matrix">input signal matrix</param>
        /// <param name="t">time</param>
        /// <returns></returns>
        protected double ContrastVision(int x, int y, double[,,] matrix, int t)
        {
            double tmp = 0;
            int r, c;
            for (int i = -1; i < 1 + 1; i++)
            {
                r = x + i;
                while (r < 0)
                    r += 1;
                while (r >= height)
                    r -= 1;
                for (int j = -1; j < 1 + 1; j++)
                {
                    c = y + j;
                    while (c < 0)
                        c += 1;
                    while (c >= width)
                        c -= 1;
                    tmp += matrix[r, c, t];
                }
            }
            return Math.Abs(matrix[x, y, t] - (tmp - matrix[x, y, t]) / 8);
        }
        /// <summary>
        /// Combine motion and contrast signals at LPTC
        /// </summary>
        /// <param name="motionSignal">motion signal from T4 and T5 neurons</param>
        /// <param name="contrastSignal">contrast signal from ON and OFF channels</param>
        /// <returns></returns>
        protected double CombineMotionAndContrast(double motionSignal, double contrastSignal)
        {
            return motionSignal - contrastSignal;
        }
        #endregion

        #region MOTION SIGNAL MAP VISUALISATION
        /// <summary>
        /// Motion signal map visualisation of T4 and T5 neurons (local motion)
        /// </summary>
        /// <param name="dir">four-cardinal direction indicator matrix</param>
        /// <param name="mag">motion magnitude matrix</param>
        /// <param name="maxMag">max motion magnitude</param>
        /// <returns></returns>
        public byte[,,] MotionSignalMap(double[,] dir, double[,] mag, double maxMag)
        {
            byte[,,] msm = new byte[height, width, 3];
            byte value;
            //assign color to each pixel
            for (int x = 0; x < height; x++)
            {
                for (int y = 0; y < width; y++)
                {
                    if (maxMag == 0)
                        value = 0;
                    else
                        //pixel intensity
                        value = (byte)(255 * mag[x, y] / maxMag);
                    //value = 255;
                    //RED for Rightward direction (degree of (-45, 45])
                    if (dir[x, y] == 0)
                    {
                        msm[x, y, 0] = 0; //B
                        msm[x, y, 1] = 0; //G
                        msm[x, y, 2] = 0; //R
                    }
                    else if (dir[x, y] >= -45 && dir[x, y] <= 45 && dir[x, y] != 0)
                    {
                        msm[x, y, 0] = 0; //B
                        msm[x, y, 1] = 0; //G
                        msm[x, y, 2] = value; //R
                    }
                    //GREEN for Downward direction (degree of (45, 135])
                    else if (dir[x, y] > 45 && dir[x, y] < 135)
                    {
                        msm[x, y, 0] = 0; //B
                        msm[x, y, 1] = value; //G
                        msm[x, y, 2] = 0; //R
                    }
                    //BLUE for Leftward direction (degree of (135, 180] & (-180, -135])
                    else if ((dir[x, y] >= 135 && dir[x, y] <= 180) || (dir[x, y] >= -180 && dir[x, y] <= -135))
                    {
                        msm[x, y, 0] = value; //B
                        msm[x, y, 1] = 0; //G
                        msm[x, y, 2] = 0; //R
                    }
                    //YELLOW for Upward direction (degree of (-135, -45])
                    else if (dir[x, y] > -135 && dir[x, y] < -45)
                    {
                        msm[x, y, 0] = 0; //B
                        msm[x, y, 1] = value; //G
                        msm[x, y, 2] = value; //R
                    }
                }
            }
            
            return msm;
        }
        #endregion

        #region NETWORK PROCESSING - LPTCs
        /// <summary>
        /// Fly motion vision neural network model for LPTC(translating) plus contrast vision
        /// </summary>
        /// <param name="img1">input grayscale image</param>
        /// <param name="img2">input grayscale image</param>
        /// <param name="t">time</param>
        public void FlyMotionVision_LPTC_And_Contrast(byte[,,] img1 /*first input image*/, byte[,,] img2 /*second input image*/, int t /*time*/)
        {
            // Init and reset some states at every time step
            int cur_t = t % 2;
            int pre_t = (t - 1) % 2;
            //double c_on = 0;
            //double c_off = 0;
            double c_on, c_off;
            LPTCs[0] = 0;
            LPTCs[1] = 0;
            LPTCs[2] = 0;
            LPTCs[3] = 0;
            LPis[0] = 0;
            LPis[1] = 0;
            LPis[2] = 0;
            LPis[3] = 0;
            SysHS = 0;
            SysVS = 0;
            //T4_maxMag = 0;
            //T5_maxMag = 0;
            double maxMotion = 0;
            double mr, ml, md, mu, vs, hs;
            // Retina layer processing
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    //Photoreceptor[y, x, cur_t] = Highpass(img1[y, x, 0], img2[y, x, 0], Photoreceptor[y, x, pre_t], delay_hp);
                    Photoreceptor[y, x, cur_t] = Highpass(img1[y, x, 0], img2[y, x, 0]);
                }
            }
            // Gaussian blur
            GB = Gaussian_Filter(Photoreceptor, Wexc, GB_ExcKernel, Winh, GB_InhKernel, cur_t);
            // ON & OFF half-wave rectifying
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    ONs[y, x, cur_t] = Halfwave_ON(GB[y, x], ONs[y, x, pre_t]);
                    OFFs[y, x, cur_t] = Halfwave_OFF(GB[y, x], OFFs[y, x, pre_t]);
                    if (blockContrast)
                    {
                        // Calculate delayed information
                        ONs_delay[y, x, cur_t] = Lowpass(ONs[y, x, cur_t], ONs_delay[y, x, pre_t], delay_lp);
                        OFFs_delay[y, x, cur_t] = Lowpass(OFFs[y, x, cur_t], OFFs_delay[y, x, pre_t], delay_lp);
                    }
                }
            }
            if (blockContrast)
            {
                // Motion correlation pathway
                for (int y = sd; y < height - sd; y++)
                {
                    for (int x = sd; x < width - sd; x++)
                    {
                        // Rightward motion
                        T4s[y, x, cur_t, 0] = Cov_TripleCorrelation(ONs[y, x, cur_t], ONs[y, x + sd, cur_t], ONs_delay[y, x, cur_t], ONs_delay[y, x + sd, cur_t]);
                        T5s[y, x, cur_t, 0] = Cov_TripleCorrelation(OFFs[y, x, cur_t], OFFs[y, x + sd, cur_t], OFFs_delay[y, x, cur_t], OFFs_delay[y, x + sd, cur_t]);
                        // Leftward motion
                        T4s[y, x, cur_t, 1] = Cov_TripleCorrelation(ONs[y, x, cur_t], ONs[y, x - sd, cur_t], ONs_delay[y, x, cur_t], ONs_delay[y, x - sd, cur_t]);
                        T5s[y, x, cur_t, 1] = Cov_TripleCorrelation(OFFs[y, x, cur_t], OFFs[y, x - sd, cur_t], OFFs_delay[y, x, cur_t], OFFs_delay[y, x - sd, cur_t]);
                        // Downward motion
                        T4s[y, x, cur_t, 2] = Cov_TripleCorrelation(ONs[y, x, cur_t], ONs[y + sd, x, cur_t], ONs_delay[y, x, cur_t], ONs_delay[y + sd, x, cur_t]);
                        T5s[y, x, cur_t, 2] = Cov_TripleCorrelation(OFFs[y, x, cur_t], OFFs[y + sd, x, cur_t], OFFs_delay[y, x, cur_t], OFFs_delay[y + sd, x, cur_t]);
                        // Upward motion
                        T4s[y, x, cur_t, 3] = Cov_TripleCorrelation(ONs[y, x, cur_t], ONs[y - sd, x, cur_t], ONs_delay[y, x, cur_t], ONs_delay[y - sd, x, cur_t]);
                        T5s[y, x, cur_t, 3] = Cov_TripleCorrelation(OFFs[y, x, cur_t], OFFs[y - sd, x, cur_t], OFFs_delay[y, x, cur_t], OFFs_delay[y - sd, x, cur_t]);
                        // Lowpass of T4 and T5 neurons
                        T4s[y, x, cur_t, 0] = Lowpass(T4s[y, x, cur_t, 0], T4s[y, x, pre_t, 0], delay_t4);
                        T4s[y, x, cur_t, 1] = Lowpass(T4s[y, x, cur_t, 1], T4s[y, x, pre_t, 1], delay_t4);
                        T4s[y, x, cur_t, 2] = Lowpass(T4s[y, x, cur_t, 2], T4s[y, x, pre_t, 2], delay_t4);
                        T4s[y, x, cur_t, 3] = Lowpass(T4s[y, x, cur_t, 3], T4s[y, x, pre_t, 3], delay_t4);
                        T5s[y, x, cur_t, 0] = Lowpass(T5s[y, x, cur_t, 0], T5s[y, x, pre_t, 0], delay_t5);
                        T5s[y, x, cur_t, 1] = Lowpass(T5s[y, x, cur_t, 1], T5s[y, x, pre_t, 1], delay_t5);
                        T5s[y, x, cur_t, 2] = Lowpass(T5s[y, x, cur_t, 2], T5s[y, x, pre_t, 2], delay_t5);
                        T5s[y, x, cur_t, 3] = Lowpass(T5s[y, x, cur_t, 3], T5s[y, x, pre_t, 3], delay_t5);
                        mr = PolarityChannelSummation(ReLU(T4s[y, x, cur_t, 0]), ReLU(T5s[y, x, cur_t, 0]), exp_ON, exp_OFF);
                        ml = PolarityChannelSummation(ReLU(T4s[y, x, cur_t, 1]), ReLU(T5s[y, x, cur_t, 1]), exp_ON, exp_OFF);
                        md = PolarityChannelSummation(ReLU(T4s[y, x, cur_t, 2]), ReLU(T5s[y, x, cur_t, 2]), exp_ON, exp_OFF);
                        mu = PolarityChannelSummation(ReLU(T4s[y, x, cur_t, 3]), ReLU(T5s[y, x, cur_t, 3]), exp_ON, exp_OFF);
                        // Local motion direction and magnitude
                        LPTCs_mag[y, x] = Math.Sqrt(Math.Pow(mr + ml, 2) + Math.Pow(md + mu, 2));
                        if (maxMotion < LPTCs_mag[y, x])
                            maxMotion = LPTCs_mag[y, x];
                        if (mr >= ml)
                            hs = mr;
                        else
                            hs = -ml;
                        if (md >= mu)
                            vs = md;
                        else
                            vs = -mu;
                        LPTCs_dir[y, x] = Math.Atan2(vs, hs) * 180 / Math.PI;
                        // LPTC combines merely motion(positive) signals
                        LPTCs[0] += (GELU_Fast(mr));
                        LPTCs[1] += (GELU_Fast(ml));
                        LPTCs[2] += (GELU_Fast(md));
                        LPTCs[3] += (GELU_Fast(mu));
                    }
                }
            }
            else
            {
                // Signal compression in ON and OFF channels
                for (int y = 0; y < height; y++)
                {
                    for (int x = 0; x < width; x++)
                    {
                        ONs_Compressed[y, x, cur_t] = normaliseContrast(Kernel_Constrast, ONs, y, x, cur_t);
                        OFFs_Compressed[y, x, cur_t] = normaliseContrast(Kernel_Constrast, OFFs, y, x, cur_t);
                        // Calculate delayed information for downstream motion correlation
                        ONs_delay[y, x, cur_t] = Lowpass(ONs_Compressed[y, x, cur_t], ONs_delay[y, x, pre_t], delay_lp);
                        OFFs_delay[y, x, cur_t] = Lowpass(OFFs_Compressed[y, x, cur_t], OFFs_delay[y, x, pre_t], delay_lp);
                    }
                }
                // Motion correlation pathway & Contrast vision pathway
                for (int y = sd; y < height - sd; y++)
                {
                    for (int x = sd; x < width - sd; x++)
                    {
                        // Contrast vision
                        ONs_Contrast[y, x, cur_t] = ContrastVision(y, x, ONs_Compressed, cur_t);
                        OFFs_Contrast[y, x, cur_t] = ContrastVision(y, x, OFFs_Compressed, cur_t);
                        c_on = Highpass(ONs_Contrast[y, x, pre_t], ONs_Contrast[y, x, cur_t]);
                        c_off = Highpass(OFFs_Contrast[y, x, pre_t], OFFs_Contrast[y, x, cur_t]);
                        //c_on = 0;
                        //c_off = 0;
                        //ONs_Contrast[y, x, cur_t] = Highpass(ONs_Contrast[y, x, pre_t], ONs_Contrast[y, x, cur_t]);
                        //OFFs_Contrast[y, x, cur_t] = Highpass(OFFs_Contrast[y, x, pre_t], OFFs_Contrast[y, x, cur_t]);
                        //c_on += ONs_Contrast[y, x, cur_t];
                        //c_off += OFFs_Contrast[y, x, cur_t];
                        // Motion correlation
                        // Rightward motion
                        T4s[y, x, cur_t, 0] = Div_TripleCorrelation(ONs_Compressed[y, x, cur_t], ONs_Compressed[y, x + sd, cur_t], ONs_delay[y, x, cur_t], ONs_delay[y, x + sd, cur_t]);
                        T5s[y, x, cur_t, 0] = Div_TripleCorrelation(OFFs_Compressed[y, x, cur_t], OFFs_Compressed[y, x + sd, cur_t], OFFs_delay[y, x, cur_t], OFFs_delay[y, x + sd, cur_t]);
                        // Leftward motion
                        T4s[y, x, cur_t, 1] = Div_TripleCorrelation(ONs_Compressed[y, x, cur_t], ONs_Compressed[y, x - sd, cur_t], ONs_delay[y, x, cur_t], ONs_delay[y, x - sd, cur_t]);
                        T5s[y, x, cur_t, 1] = Div_TripleCorrelation(OFFs_Compressed[y, x, cur_t], OFFs_Compressed[y, x - sd, cur_t], OFFs_delay[y, x, cur_t], OFFs_delay[y, x - sd, cur_t]);
                        // Downward motion
                        T4s[y, x, cur_t, 2] = Div_TripleCorrelation(ONs_Compressed[y, x, cur_t], ONs_Compressed[y + sd, x, cur_t], ONs_delay[y, x, cur_t], ONs_delay[y + sd, x, cur_t]);
                        T5s[y, x, cur_t, 2] = Div_TripleCorrelation(OFFs_Compressed[y, x, cur_t], OFFs_Compressed[y + sd, x, cur_t], OFFs_delay[y, x, cur_t], OFFs_delay[y + sd, x, cur_t]);
                        // Upward motion
                        T4s[y, x, cur_t, 3] = Div_TripleCorrelation(ONs_Compressed[y, x, cur_t], ONs_Compressed[y - sd, x, cur_t], ONs_delay[y, x, cur_t], ONs_delay[y - sd, x, cur_t]);
                        T5s[y, x, cur_t, 3] = Div_TripleCorrelation(OFFs_Compressed[y, x, cur_t], OFFs_Compressed[y - sd, x, cur_t], OFFs_delay[y, x, cur_t], OFFs_delay[y - sd, x, cur_t]);
                        // Lowpass of T4 and T5 neurons
                        T4s[y, x, cur_t, 0] = Lowpass(T4s[y, x, cur_t, 0], T4s[y, x, pre_t, 0], delay_t4);
                        T4s[y, x, cur_t, 1] = Lowpass(T4s[y, x, cur_t, 1], T4s[y, x, pre_t, 1], delay_t4);
                        T4s[y, x, cur_t, 2] = Lowpass(T4s[y, x, cur_t, 2], T4s[y, x, pre_t, 2], delay_t4);
                        T4s[y, x, cur_t, 3] = Lowpass(T4s[y, x, cur_t, 3], T4s[y, x, pre_t, 3], delay_t4);
                        T5s[y, x, cur_t, 0] = Lowpass(T5s[y, x, cur_t, 0], T5s[y, x, pre_t, 0], delay_t5);
                        T5s[y, x, cur_t, 1] = Lowpass(T5s[y, x, cur_t, 1], T5s[y, x, pre_t, 1], delay_t5);
                        T5s[y, x, cur_t, 2] = Lowpass(T5s[y, x, cur_t, 2], T5s[y, x, pre_t, 2], delay_t5);
                        T5s[y, x, cur_t, 3] = Lowpass(T5s[y, x, cur_t, 3], T5s[y, x, pre_t, 3], delay_t5);
                        mr = PolarityChannelSummation(ReLU(T4s[y, x, cur_t, 0] * coe_motion - c_on * coe_contrast), ReLU(T5s[y, x, cur_t, 0] * coe_motion - c_off * coe_contrast), exp_ON, exp_OFF);
                        ml = PolarityChannelSummation(ReLU(T4s[y, x, cur_t, 1] * coe_motion - c_on * coe_contrast), ReLU(T5s[y, x, cur_t, 1] * coe_motion - c_off * coe_contrast), exp_ON, exp_OFF);
                        md = PolarityChannelSummation(ReLU(T4s[y, x, cur_t, 2] * coe_motion - c_on * coe_contrast), ReLU(T5s[y, x, cur_t, 2] * coe_motion - c_off * coe_contrast), exp_ON, exp_OFF);
                        mu = PolarityChannelSummation(ReLU(T4s[y, x, cur_t, 3] * coe_motion - c_on * coe_contrast), ReLU(T5s[y, x, cur_t, 3] * coe_motion - c_off * coe_contrast), exp_ON, exp_OFF);
                        // Local motion direction and magnitude
                        LPTCs_mag[y, x] = Math.Sqrt(Math.Pow(mr + ml, 2) + Math.Pow(md + mu, 2));
                        if (maxMotion < LPTCs_mag[y, x])
                            maxMotion = LPTCs_mag[y, x];
                        if (mr >= ml)
                            hs = mr;
                        else
                            hs = -ml;
                        if (md >= mu)
                            vs = md;
                        else
                            vs = -mu;
                        LPTCs_dir[y, x] = Math.Atan2(vs, hs) * 180 / Math.PI;
                        // LPTC combines merely motion(positive) signals
                        LPTCs[0] += (GELU_Fast(mr));
                        LPTCs[1] += (GELU_Fast(ml));
                        LPTCs[2] += (GELU_Fast(md));
                        LPTCs[3] += (GELU_Fast(mu));
                        // LPTC combines motion(positive) and contrast(negative) signals
                        //LPTCs[0] += (GELU_Precise(PolarityChannelSummation(ReLU(T4s[y, x, cur_t, 0] - ONs_Contrast[y, x, cur_t]), ReLU(T5s[y, x, cur_t, 0] - OFFs_Contrast[y, x, cur_t]), exp_ON, exp_OFF)));
                        //LPTCs[1] += (GELU_Precise(PolarityChannelSummation(ReLU(T4s[y, x, cur_t, 1] - ONs_Contrast[y, x, cur_t]), ReLU(T5s[y, x, cur_t, 1] - OFFs_Contrast[y, x, cur_t]), exp_ON, exp_OFF)));
                        //LPTCs[2] += (GELU_Precise(PolarityChannelSummation(ReLU(T4s[y, x, cur_t, 2] - ONs_Contrast[y, x, cur_t]), ReLU(T5s[y, x, cur_t, 2] - OFFs_Contrast[y, x, cur_t]), exp_ON, exp_OFF)));
                        //LPTCs[3] += (GELU_Precise(PolarityChannelSummation(ReLU(T4s[y, x, cur_t, 3] - ONs_Contrast[y, x, cur_t]), ReLU(T5s[y, x, cur_t, 3] - OFFs_Contrast[y, x, cur_t]), exp_ON, exp_OFF)));
                        //LPTCs[0] += (GELU_Precise(PolarityChannelSummation(ReLU(T4s[y, x, cur_t, 0] * coe_motion - c_on * coe_contrast), ReLU(T5s[y, x, cur_t, 0] * coe_motion - c_off * coe_contrast), exp_ON, exp_OFF)));
                        //LPTCs[1] += (GELU_Precise(PolarityChannelSummation(ReLU(T4s[y, x, cur_t, 1] * coe_motion - c_on * coe_contrast), ReLU(T5s[y, x, cur_t, 1] * coe_motion - c_off * coe_contrast), exp_ON, exp_OFF)));
                        //LPTCs[2] += (GELU_Precise(PolarityChannelSummation(ReLU(T4s[y, x, cur_t, 2] * coe_motion - c_on * coe_contrast), ReLU(T5s[y, x, cur_t, 2] * coe_motion - c_off * coe_contrast), exp_ON, exp_OFF)));
                        //LPTCs[3] += (GELU_Precise(PolarityChannelSummation(ReLU(T4s[y, x, cur_t, 3] * coe_motion - c_on * coe_contrast), ReLU(T5s[y, x, cur_t, 3] * coe_motion - c_off * coe_contrast), exp_ON, exp_OFF)));
                    }
                }
            }
            // LPTC-LPi
            LPis[0] = LPTCs[1];
            LPis[1] = LPTCs[0];
            LPis[2] = LPTCs[3];
            LPis[3] = LPTCs[2];
            LPTCs[0] = ReLU(LPTCs[0] - LPis[0]);
            LPTCs[1] = ReLU(LPTCs[1] - LPis[1]);
            LPTCs[2] = ReLU(LPTCs[2] - LPis[2]);
            LPTCs[3] = ReLU(LPTCs[3] - LPis[3]);
            SysHS = LPTCs[0] - LPTCs[1];
            SysVS = LPTCs[2] - LPTCs[3];
            // Motion signal maps
            LPTCs_msm = MotionSignalMap(LPTCs_dir, LPTCs_mag, maxMotion);

            //Console.WriteLine("Frame  {0}  R  {1:F}  L  {2:F}  D  {3:F}  U  {4:F}  ON-Contrast  {5:F}  OFF-Contrast  {6:F}  HS  {7:F}  VS  {8:F}", t, LPTCs[0], LPTCs[1], LPTCs[2], LPTCs[3], c_on, c_off, SysHS, SysVS);
            Console.WriteLine("Frame  {0}  R  {1:F}  L  {2:F}  D  {3:F}  U  {4:F}  HS  {5:F}  VS  {6:F}", t, LPTCs[0], LPTCs[1], LPTCs[2], LPTCs[3], SysHS, SysVS);
        }
        #endregion

        #region NETWORK PROCESSING - LPLC2
        /// <summary>
        /// Fly early visual processing neural network model for LPLC2 (ultra-selective looming)
        /// </summary>
        /// <param name="img1">input grayscale image</param>
        /// <param name="img2">input grayscale image</param>
        /// <param name="t">time</param>
        public void FlyMotionVision_LPLC2(byte[,,] img1 /*first input image*/, byte[,,] img2 /*second input image*/, int t /*time*/)
        {
            // Init and reset some states at every time step
            double mr, ml, md, mu, vs, hs;
            int cur_t = t % 2;
            int pre_t = (t - 1) % 2;
            int cur_avg = t % time_length;
            double c_on, c_off;
            LPLC2_Region[0] = 0;
            LPLC2_Region[1] = 0;
            LPLC2_Region[2] = 0;
            LPLC2_Region[3] = 0;
            double maxMotion = 0;
            double tmpSum = 0;
            // Retina layer processing
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    //Photoreceptor[y, x, cur_t] = Highpass(img1[y, x, 0], img2[y, x, 0]);
                    Photoreceptor[y, x, cur_t] = Highpass(img1[y, x, 0], img2[y, x, 0], Photoreceptor[y, x, pre_t], delay_hp);
                }
            }
            // Gaussian blur
            GB = Gaussian_Filter(Photoreceptor, Wexc, GB_ExcKernel, Winh, GB_InhKernel, cur_t);
            // ON & OFF half-wave rectifying
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    ONs[y, x, cur_t] = Halfwave_ON(GB[y, x], ONs[y, x, pre_t]);
                    OFFs[y, x, cur_t] = Halfwave_OFF(GB[y, x], OFFs[y, x, pre_t]);
                }
            }
            // Signal compression in ON and OFF channels
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    // Tanh : 
                    ONs_Compressed[y, x, cur_t] = normaliseContrast(Kernel_Constrast, ONs, y, x, cur_t); // to constrain value within 0 ~ 1 
                    OFFs_Compressed[y, x, cur_t] = normaliseContrast(Kernel_Constrast, OFFs, y, x, cur_t);
                    // Calculate delayed information for downstream motion correlation
                    ONs_delay[y, x, cur_t] = Lowpass(ONs_Compressed[y, x, cur_t], ONs_delay[y, x, pre_t], delay_lp);
                    OFFs_delay[y, x, cur_t] = Lowpass(OFFs_Compressed[y, x, cur_t], OFFs_delay[y, x, pre_t], delay_lp);
                }
            }
            // Motion correlation pathway & Contrast vision pathway
            for (int y = sd; y < height - sd; y++)
            {
                for (int x = sd; x < width - sd; x++)
                {
                    // Contrast vision
                    ONs_Contrast[y, x, cur_t] = ContrastVision(y, x, ONs_Compressed, cur_t);
                    OFFs_Contrast[y, x, cur_t] = ContrastVision(y, x, OFFs_Compressed, cur_t);
                    c_on = Highpass(ONs_Contrast[y, x, pre_t], ONs_Contrast[y, x, cur_t]);
                    c_off = Highpass(OFFs_Contrast[y, x, pre_t], OFFs_Contrast[y, x, cur_t]);
                    // Motion correlation
                    // Rightward motion
                    T4s[y, x, cur_t, 0] = Div_TripleCorrelation_Bias(ONs_Compressed[y, x, cur_t], ONs_Compressed[y, x + sd, cur_t], ONs_delay[y, x, cur_t], ONs_delay[y, x + sd, cur_t], Radial_Bias[y, x]);
                    T5s[y, x, cur_t, 0] = Div_TripleCorrelation_Bias(OFFs_Compressed[y, x, cur_t], OFFs_Compressed[y, x + sd, cur_t], OFFs_delay[y, x, cur_t], OFFs_delay[y, x + sd, cur_t], Radial_Bias[y, x]);
                    // Leftward motion
                    T4s[y, x, cur_t, 1] = Div_TripleCorrelation_Bias(ONs_Compressed[y, x, cur_t], ONs_Compressed[y, x - sd, cur_t], ONs_delay[y, x, cur_t], ONs_delay[y, x - sd, cur_t], Radial_Bias[y, x]);
                    T5s[y, x, cur_t, 1] = Div_TripleCorrelation_Bias(OFFs_Compressed[y, x, cur_t], OFFs_Compressed[y, x - sd, cur_t], OFFs_delay[y, x, cur_t], OFFs_delay[y, x - sd, cur_t], Radial_Bias[y, x]);
                    // Downward motion
                    T4s[y, x, cur_t, 2] = Div_TripleCorrelation_Bias(ONs_Compressed[y, x, cur_t], ONs_Compressed[y + sd, x, cur_t], ONs_delay[y, x, cur_t], ONs_delay[y + sd, x, cur_t], Radial_Bias[y, x]);
                    T5s[y, x, cur_t, 2] = Div_TripleCorrelation_Bias(OFFs_Compressed[y, x, cur_t], OFFs_Compressed[y + sd, x, cur_t], OFFs_delay[y, x, cur_t], OFFs_delay[y + sd, x, cur_t], Radial_Bias[y, x]);
                    // Upward motion
                    T4s[y, x, cur_t, 3] = Div_TripleCorrelation_Bias(ONs_Compressed[y, x, cur_t], ONs_Compressed[y - sd, x, cur_t], ONs_delay[y, x, cur_t], ONs_delay[y - sd, x, cur_t], Radial_Bias[y, x]);
                    T5s[y, x, cur_t, 3] = Div_TripleCorrelation_Bias(OFFs_Compressed[y, x, cur_t], OFFs_Compressed[y - sd, x, cur_t], OFFs_delay[y, x, cur_t], OFFs_delay[y - sd, x, cur_t], Radial_Bias[y, x]);
                    // Lowpass of T4 and T5 neurons
                    T4s[y, x, cur_t, 0] = Lowpass(T4s[y, x, cur_t, 0], T4s[y, x, pre_t, 0], delay_t4);
                    T4s[y, x, cur_t, 1] = Lowpass(T4s[y, x, cur_t, 1], T4s[y, x, pre_t, 1], delay_t4);
                    T4s[y, x, cur_t, 2] = Lowpass(T4s[y, x, cur_t, 2], T4s[y, x, pre_t, 2], delay_t4);
                    T4s[y, x, cur_t, 3] = Lowpass(T4s[y, x, cur_t, 3], T4s[y, x, pre_t, 3], delay_t4);
                    T5s[y, x, cur_t, 0] = Lowpass(T5s[y, x, cur_t, 0], T5s[y, x, pre_t, 0], delay_t5);
                    T5s[y, x, cur_t, 1] = Lowpass(T5s[y, x, cur_t, 1], T5s[y, x, pre_t, 1], delay_t5);
                    T5s[y, x, cur_t, 2] = Lowpass(T5s[y, x, cur_t, 2], T5s[y, x, pre_t, 2], delay_t5);
                    T5s[y, x, cur_t, 3] = Lowpass(T5s[y, x, cur_t, 3], T5s[y, x, pre_t, 3], delay_t5);
                    mr = PolarityChannelSummation(ReLU(T4s[y, x, cur_t, 0] * coe_motion - c_on * coe_contrast), ReLU(T5s[y, x, cur_t, 0] * coe_motion - c_off * coe_contrast), exp_ON, exp_OFF);
                    ml = PolarityChannelSummation(ReLU(T4s[y, x, cur_t, 1] * coe_motion - c_on * coe_contrast), ReLU(T5s[y, x, cur_t, 1] * coe_motion - c_off * coe_contrast), exp_ON, exp_OFF);
                    md = PolarityChannelSummation(ReLU(T4s[y, x, cur_t, 2] * coe_motion - c_on * coe_contrast), ReLU(T5s[y, x, cur_t, 2] * coe_motion - c_off * coe_contrast), exp_ON, exp_OFF);
                    mu = PolarityChannelSummation(ReLU(T4s[y, x, cur_t, 3] * coe_motion - c_on * coe_contrast), ReLU(T5s[y, x, cur_t, 3] * coe_motion - c_off * coe_contrast), exp_ON, exp_OFF);
                    // Local motion direction and magnitude
                    LPLC2_mag[y, x] = Math.Sqrt(Math.Pow(mr + ml, 2) + Math.Pow(md + mu, 2));
                    if (maxMotion < LPLC2_mag[y, x])
                        maxMotion = LPLC2_mag[y, x];
                    if (mr >= ml)
                        hs = mr;
                    else
                        hs = -ml;
                    if (md >= mu)
                        vs = md;
                    else
                        vs = -mu;
                    LPLC2_dir[y, x] = Math.Atan2(vs, hs) * 180 / Math.PI;
                    // Regional integration of visual projection neurons
                    if (y <= View_Centroid[0] && x <= View_Centroid[1])         // Left-up region
                    {
                        LPLC2_Region[0] += GELU_Precise(ml - mr);
                        LPLC2_Region[0] += GELU_Precise(mu - md);
                    }
                    else if (y <= View_Centroid[0] && x >= View_Centroid[1])    // Right-up region
                    {
                        LPLC2_Region[1] += GELU_Precise(mr - ml);
                        LPLC2_Region[1] += GELU_Precise(mu - md);
                    }
                    else if (y >= View_Centroid[0] && x <= View_Centroid[1])    // Left-bottom region
                    {
                        LPLC2_Region[2] += GELU_Precise(ml - mr);
                        LPLC2_Region[2] += GELU_Precise(md - mu);
                    }
                    else if (y >= View_Centroid[0] && x >= View_Centroid[1])    // Right-bottom region
                    {
                        LPLC2_Region[3] += GELU_Precise(mr - ml);
                        LPLC2_Region[3] += GELU_Precise(md - mu);
                    }
                }
            }
            // Looming response of population LPLC2
            LPLC2[cur_avg] = ReLU(LPLC2_Region[0]) * ReLU(LPLC2_Region[1]) * ReLU(LPLC2_Region[2]) * ReLU(LPLC2_Region[3]);
            if (LPLC2[cur_avg] > 0)
            {
                for (int i = 0; i < time_length; i++)
                {
                    tmpSum += LPLC2[i];
                }
                LPLC2_Out = tmpSum / time_length;
            }
            else
                LPLC2_Out = 0;
            // Motion signal maps
            LPLC2_msm = MotionSignalMap(LPLC2_dir, LPLC2_mag, maxMotion);


            Console.WriteLine("Frame  {0}  Lu  {1:F}  Ru  {2:F}  Lb  {3:F}  Rb  {4:F}  LPLC2  {5:F}", t, LPLC2_Region[0], LPLC2_Region[1], LPLC2_Region[2], LPLC2_Region[3], LPLC2_Out);
        }
        #endregion

        #region NETWORK PROCESSING - FOREGROUND & BACKGROUND TRANSLATING DECODING
        #endregion
    }
}

