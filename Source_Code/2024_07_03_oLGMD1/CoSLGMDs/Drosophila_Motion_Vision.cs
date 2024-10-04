/*
 * Filename: Drosophila_Motion_Vision.cs
 * Date: 2019-2020
 * Author: Dr Qinbing Fu
 * Institution: Guangzhou University; University of Lincoln
 * Contact: qinbingfu87@gmail.com
 * Google Scholar Webpage: https://scholar.google.com/citations?user=YIte1M8AAAAJ&hl=en
 * ResearchGate Webpage: https://www.researchgate.net/profile/Qinbing-Fu-2
 */


using System;


namespace EMD
{
    /// <summary>
    /// This C# class is used as open source code to represent the algorithms on implementing Drosophila motion vision pathway neural network model for decoding foreground translating object in front of cluttered moving background.
    /// This plain code mainly runs on Windows system platform that can be modified to C, C++, Python, Java fitting with other systems.
    /// </summary>
    internal class Drosophila_Motion_Vision
    {
        #region FIELD
        /// <summary>
        /// Sampling distance (1~16 pixels)
        /// </summary>
        protected readonly int d;
        /// <summary>
        /// Delay coefficient in output T4/T5 EMDs
        /// </summary>
        protected readonly double emd_t;
        /// <summary>
        /// Image width
        /// </summary>
        protected readonly int width;
        /// <summary>
        /// Image height
        /// </summary>
        protected readonly int height;
        /// <summary>
        /// Horizontal system matrix of EMDs(T4/T5), dimension: height x width x 2(time)
        /// </summary>
        protected double[,,] emd_h;
        /// <summary>
        /// Vertical system of EMDs(T4/T5), dimension: height x width x 2(time)
        /// </summary>
        protected double[,,] emd_v;
        /// <summary>
        /// Non-linear processing unit in the motion correlator
        /// </summary>
        protected double mlp1;
        /// <summary>
        /// Non-linear processing unit in the motion correlator
        /// </summary>
        protected double mlp2;
        /// <summary>
        /// Temporal subtractor in L1 pathway
        /// </summary>
        protected double on_on_sub;
        /// <summary>
        /// Temporal subtractor in L2 pathway
        /// </summary>
        protected double off_off_sub;
        /// <summary>
        /// Max rgb value: 255
        /// </summary>
        protected  readonly byte maxRgbValue;
        /// <summary>
        /// Max local motion magnitude in each frame processing
        /// </summary>
        protected double maxMag;
        /// <summary>
        /// Rxcitory standard deviation in the DoGs (Difference of Gaussians) algorithm
        /// </summary>
        protected readonly double Esd;
        /// <summary>
        /// Inhibitory standard deviation in the DoGs (Difference of Gaussians) algorithm
        /// </summary>
        protected readonly double Isd;
        /// <summary>
        /// Inhibitory(negative) gaussian kernel width in DoGs, indicating the size of receiptive field, normally being set to twice the samling distance
        /// </summary>
        protected readonly int inh_gauss_width;
        /// <summary>
        /// Excitatory(positive) gaussian kernel width in DoGs, normally with the half size of inhibitory(negative) gaussian kernel
        /// </summary>
        protected readonly int exc_gauss_width;
        /// <summary>
        /// Broader(outer) inhibitory gaussian convolutional kernel
        /// </summary>
        protected double[,] inh_gauss_kernel;
        /// <summary>
        /// Inner excitatory gaussian convolutional kernel
        /// </summary>
        protected double[,] exc_gauss_kernel;
        /// <summary>
        /// Number of adjacent and connected ON/OFF interneurons
        /// </summary>
        protected readonly byte Ncon;
        /// <summary>
        /// Increasing step in sampling distance
        /// </summary>
        protected readonly byte Ncon_step;
        /// <summary>
        /// Clip point in the half-wave rectifiers of entries to ON and OFF pathways
        /// </summary>
        protected readonly double clip_point;
        /// <summary>
        /// Dynamic time params in low-pass filter varied by the sampling distance
        /// </summary>
        protected int[] dyn_lp_tau;
        /// <summary>
        /// Dynamic time constant in low-pass filter calculated by tau and discrete time interval
        /// </summary>
        protected double[] dyn_lp_delay;
        /// <summary>
        /// Frames per second of input stimuli video
        /// </summary>
        protected readonly int fps;
        /// <summary>
        /// Coefficient coe_a * [ON] for L1 output in local motion detector combining L1 with L2
        /// {0 ~ 6}
        /// </summary>
        protected readonly int coe_a;
        /// <summary>
        /// Coefficient coe_b * [OFF] for L2 output in local motion detector combining L1 with L2
        /// {0 ~ 6}
        /// </summary>
        protected readonly int coe_b;
        /// <summary>
        /// Coefficient coe_c * [ON] * [OFF]
        /// {0 ~ 6}
        /// </summary>
        protected readonly int coe_c;
        /// <summary>
        /// Bias factor in inhibitory flows
        /// </summary>
        protected readonly double inh_bias;
        /// <summary>
        /// Small positive number to scale sigmoid transfer
        /// </summary>
        protected readonly double delta_C;
        /// <summary>
        /// Small real number to scale sigmoid transfer 
        /// </summary>
        protected readonly double k;
        /// <summary>
        /// Smaller time constant in FDSR filter
        /// </summary>
        protected readonly int fdsr_fast_tau;
        /// <summary>
        /// Larger time constant in FDSR filter
        /// </summary>
        protected readonly int fdsr_slow_tau;
        /// <summary>
        /// Shorter time delay in FDSR mechanism
        /// </summary>
        protected readonly double fdsr_fast_delay;
        /// <summary>
        /// longer time delay in FDSR mechanism
        /// </summary>
        protected readonly double fdsr_slow_delay;
        /// <summary>
        /// Number of cells in retina layer
        /// </summary>
        protected readonly int Ncell;
        /// <summary>
        /// Time interval in milliseconds between discrete signal frames
        /// </summary>
        protected readonly double time_interval;
        #endregion

        #region NETWORK LAYERS
        /// <summary>
        /// High-pass filtered photoreceptors matrix, dimension: height x width x 2(time)
        /// </summary>
        protected int[,,] HPF;
        /// <summary>
        /// ON interneurons matrix, dimension: height x width x 2(time)
        /// </summary>
        protected double[,,] ONs;
        /// <summary>
        /// OFF interneurons matrix, dimension: height x width x 2(time)
        /// </summary>
        protected double[,,] OFFs;
        /// <summary>
        /// First-order low-pass filtered signal matrix in ON channels, dimension: height x width
        /// </summary>
        protected double[,] ON_lpf;
        /// <summary>
        /// First-order low-pass filtered signal matrix in OFF channels, dimension: height x width
        /// </summary>
        protected double[,] OFF_lpf;
        /// <summary>
        /// Horizontal sensitive T4 cells, dimension: height x width x 2(time)
        /// </summary>
        protected double[,,] ON_ON_hor;
        /// <summary>
        /// Vertical sensitive T4 cells, dimension: height x width x 2(time)
        /// </summary>
        protected double[,,] ON_ON_ver;
        /// <summary>
        /// Horizontal sensitive T5 cells, dimension: height x width x 2(time)
        /// </summary>
        protected double[,,] OFF_OFF_hor;
        /// <summary>
        /// Vertical sensitive T5 cells, dimension: height x width x 2(time)
        /// </summary>
        protected double[,,] OFF_OFF_ver;
        /// <summary>
        /// ON-ON horizontal wide field motion in time series
        /// </summary>
        protected double wfm_on_on_hor;
        /// <summary>
        /// OFF-OFF horizontal wide field motion in time series
        /// </summary>
        protected double wfm_off_off_hor;
        /// <summary>
        /// ON-ON vertical wide field motion in time series
        /// </summary>
        protected double wfm_on_on_ver;
        /// <summary>
        /// OFF-OFF vertical wide field motion in time series
        /// </summary>
        protected double wfm_off_off_ver;
        /// <summary>
        /// Horizontal system of wide field motion detectors
        /// </summary>
        protected double wfm_h;
        /// <summary>
        /// Vertical system of wide field motion detectors
        /// </summary>
        protected double wfm_v;
        /// <summary>
        /// Horizontal system of visual odometer
        /// </summary>
        protected double vo_h;
        /// <summary>
        /// Vertical system of visual odometer
        /// </summary>
        protected double vo_v;
        /// <summary>
        /// Local motion magnitude matrix, dimension: height x width x 2(time)
        /// </summary>
        protected double[,,] mag;
        /// <summary>
        /// Local motion direction matrix, dimension: height x width x 2(time)
        /// </summary>
        protected double[,,] dir;
        /// <summary>
        /// Motion signal map (RGB) matrix, dimension: height x width x 3(RGB)
        /// </summary>
        protected byte[,,] msm;
        #endregion

        #region CONTRUCTOR
        /// <summary>
        /// Parametric Constructor
        /// </summary>
        /// <param name="w">image width</param>
        /// <param name="h">image height</param>
        /// <param name="fps">frames per second</param>
        public Drosophila_Motion_Vision(int w, int h, int fps)
        {
            Ncon = 4;
            Ncon_step = 2;
            inh_bias = 1;
            coe_a = 1;
            coe_b = 1;
            coe_c = 1;
            width = w;
            height = h;
            Ncell = w * h;
            d = 2 * Ncon_step;
            Esd = 0.5 * d;
            Isd = d;
            inh_gauss_width = 2 * d + 1;
            exc_gauss_width = d + 1;
            emd_t = 0.9;
            this.fps = fps;
            time_interval = 1000 / this.fps; //ms
            maxRgbValue = 255;
            maxMag = 0;
            inh_gauss_kernel = new double[inh_gauss_width, inh_gauss_width];
            exc_gauss_kernel = new double[exc_gauss_width, exc_gauss_width];
            //attention
            emd_h = new double[height, width, 2];
            emd_v = new double[height, width, 2];
            HPF = new int[height, width, 2];
            msm = new byte[height, width, 3];
            wfm_h = 0;
            wfm_v = 0;
            mag = new double[height, width, 2];
            dir = new double[height, width, 2];
            vo_h = 0;
            vo_v = 0;
            clip_point = 0;
            delta_C = 0.5;
            k = 0.01;
            fdsr_fast_tau = 1; //ms
            fdsr_slow_tau = 100; //ms
            dyn_lp_delay = new double[Ncon - 1];
            dyn_lp_tau = new int[Ncon - 1];
            dyn_lp_tau[0] = 200; //ms
            dyn_lp_delay[0] = time_interval / (time_interval + dyn_lp_tau[0]);
            for (int i = 1; i < Ncon - 1; i++)
            {
                dyn_lp_tau[i] = dyn_lp_tau[0] - i * 40; //ms
                if (dyn_lp_tau[i] <= 10)
                    dyn_lp_tau[i] = 10; //ms
                dyn_lp_delay[i] = time_interval / (time_interval + dyn_lp_tau[i]);
            }
            fdsr_fast_delay = time_interval / (time_interval + fdsr_fast_tau);
            fdsr_slow_delay = time_interval / (time_interval + fdsr_slow_tau);
            ONs = new double[height, width, 2];
            OFFs = new double[height, width, 2];
            ON_lpf = new double[height, width];
            OFF_lpf = new double[height, width];
            ON_ON_hor = new double[height, width, 2];
            ON_ON_ver = new double[height, width, 2];
            OFF_OFF_hor = new double[height, width, 2];
            OFF_OFF_ver = new double[height, width, 2];
            //Make Gaussian Kernels
            MakeGaussian(Esd, exc_gauss_width, ref exc_gauss_kernel);
            MakeGaussian(Isd, inh_gauss_width, ref inh_gauss_kernel);


            Console.WriteLine("Drosophila motion vision neural network model parameters setting ready........\n\n");
        }
        #endregion

        #region METHOD
        /// <summary>
        /// Make Gaussian kernel
        /// </summary>
        /// <param name="sigma">standard deviation</param>
        /// <param name="gauss_width">width of gauss</param>
        /// <param name="gauss_kernel">gauss kernel</param>
        protected void MakeGaussian(double sigma, int gauss_width, ref double[,] gauss_kernel)
        {
            //coordinate of center point
            int center_x = gauss_width / 2;
            int center_y = gauss_width / 2;

            double gaussian; //record different gaussian value according to different sigma
            double distance;
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
        /// First-order high-pass filter
        /// </summary>
        /// <param name="in_val1">input signal value</param>
        /// <param name="in_val2">input signal value</param>
        /// <returns></returns>
        protected int HighpassFilter(byte in_val1, byte in_val2)
        {
            return in_val2 - in_val1;
        }
        /// <summary>
        /// Difference of Gaussians algorithm
        /// </summary>
        /// <param name="input_image">input image matrix</param>
        /// <param name="t">timestamp</param>
        /// <returns></returns>
        protected int[,] DOG(int[,,] input_image, int t)
        {
            //first convolution
            int[,] tmp1 = Average_Filter(input_image, exc_gauss_width, exc_gauss_kernel, t);
            //second convolution
            int[,] tmp2 = Average_Filter(input_image, inh_gauss_width, inh_gauss_kernel, t);
            //difference of gaussians
            return Sub_Images(tmp1, tmp2);
        }
        /// <summary>
        /// Gaussian convolution
        /// </summary>
        /// <param name="input">input image array</param>
        /// <param name="gauss_width">width of gauss</param>
        /// <param name="gauss_kernel">gauss kernel</param>
        /// <param name="t"></param>
        /// <returns></returns>
        protected int[,] Average_Filter(int[,,] input, int gauss_width, double[,] gauss_kernel, int t)
        {
            int[,] output = new int[height, width];
            int tmp = 0;
            //kernel radius
            int k_radius = gauss_width / 2;
            int r, c;
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    for (int i = -k_radius; i < k_radius + 1; i++)
                    {
                        for (int j = -k_radius; j < k_radius + 1; j++)
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
                            tmp = (int)(input[r, c, t] * gauss_kernel[i + k_radius, j + k_radius]);
                            output[y, x] += tmp;
                        }
                    }
                }
            }
            return output;
        }
        /// <summary>
        /// Subtraction in DoGs with polarity selectivity
        /// </summary>
        /// <param name="first_image">input image array after being convolved by the excitatory gausssian kernel</param>
        /// <param name="second_image">input image array after being convolved by the inhibitory gausssian kernel</param>
        /// <returns></returns>
        protected int[,] Sub_Images(int[,] first_image, int[,] second_image)
        {
            int[,] result_image = new int[height, width];

            for (int i = 0; i < height; i++)
            {
                for (int j = 0; j < width; j++)
                {
                    result_image[i, j] = first_image[i, j] - second_image[i, j];

                    if (first_image[i, j] >= 0 && second_image[i, j] >= 0)
                    {
                        result_image[i, j] = Math.Abs(result_image[i, j]);
                    }
                    else if (first_image[i, j] <= 0 && second_image[i, j] <= 0)
                    {
                        if (result_image[i, j] > 0)
                            result_image[i, j] = -result_image[i, j];
                    }

                }
            }
            return result_image;
        }
        /// <summary>
        /// Half-wave rectifier
        /// </summary>
        /// <param name="input">input grayscale value</param>
        /// <param name="x">abscissa</param>
        /// <param name="y">ordinate</param>
        /// <param name="t">current time step</param>
        protected void HalfWaveRectifier(int input, int x, int y, int t)
        {
            if (input >= clip_point)
            {
                ONs[x, y, t] = input;
                OFFs[x, y, t] = 0;
            }
            else
            {
                OFFs[x, y, t] = Math.Abs(input);
                ONs[x, y, t] = 0;
            }
        }
        /// <summary>
        /// Fast depolarizing slow repolarizing
        /// </summary>
        /// <param name="pre_on">previous ON value</param>
        /// <param name="cur_on">current ON value</param>
        /// <param name="pre_off">previous OFF value</param>
        /// <param name="cur_off">current OFF value</param>
        protected void FDSR(double pre_on, ref double cur_on, double pre_off, ref double cur_off)
        {
            //calculate the change gradients
            double on_grad = cur_on - pre_on;
            double off_grad = cur_off - pre_off;
            //fast onset slow decay
            if (on_grad >= 0)
                cur_on = cur_on - LowpassFilter(cur_on, pre_on, fdsr_fast_delay);
            else
                cur_on = cur_on - LowpassFilter(cur_on, pre_on, fdsr_slow_delay);
            if (off_grad >= 0)
                cur_off = cur_off - LowpassFilter(cur_off, pre_off, fdsr_fast_delay);
            else
                cur_off = cur_off - LowpassFilter(cur_off, pre_off, fdsr_slow_delay);
            //positive outputs
            if (cur_on < 0)
                cur_on = 0;
            if (cur_off < 0)
                cur_off = 0;
        }
        /// <summary>
        /// Low-pass filter
        /// </summary>
        /// <param name="cur_input">current input signal</param>
        /// <param name="pre_input">previous input signal</param>
        /// <param name="lp_delay">low-pass delay coefficient</param>
        /// <returns></returns>
        protected double LowpassFilter(double cur_input, double pre_input, double lp_delay)
        {
            return lp_delay * cur_input + (1 - lp_delay) * pre_input;
        }
        /// <summary>
        /// Sigmoid transformation - activation function
        /// </summary>
        /// <param name="Kf">membrane potential</param>
        /// <returns></returns>
        protected double SigmoidTransfer(double Kf)
        {
            return Math.Pow(1 + Math.Exp(-Kf * Math.Pow(Ncell * k, -1)), -1);
        }
        /// <summary>
        /// Draw motion signal map of current frame (t) with four cardinal directions (R, L, U, D)
        /// Defining color indicators: red for front-to-back (rightwards), blue for back-to-front (leftwards), green for upwards, yellow for downwards
        /// </summary>
        /// <param name="t">current time step</param>
        protected void MotionSM(int t)
        {
            byte value = 0;
            int cur_frame = t % 2;
            //assign color to each pixel
            for (int x = 0; x < height; x++)
            {
                for (int y = 0; y < width; y++)
                {
                    //pixel intensity
                    value = (byte)(maxRgbValue * mag[x, y, cur_frame] / maxMag);
                    //value = 255;
                    //RED for Rightward direction (degree of (-45, 45])
                    if (dir[x, y, cur_frame] == 0)
                    {
                        msm[x, y, 0] = 0; //B
                        msm[x, y, 1] = 0; //G
                        msm[x, y, 2] = 0; //R
                    }
                    else if (dir[x, y, cur_frame] >= -45 && dir[x, y, cur_frame] <= 45 && dir[x, y, cur_frame] != 0)
                    {
                        msm[x, y, 0] = 0; //B
                        msm[x, y, 1] = 0; //G
                        msm[x, y, 2] = value; //R
                    }
                    //GREEN for Downward direction (degree of (45, 135])
                    else if (dir[x, y, cur_frame] > 45 && dir[x, y, cur_frame] < 135)
                    {
                        msm[x, y, 0] = 0; //B
                        msm[x, y, 1] = value; //G
                        msm[x, y, 2] = 0; //R
                    }
                    //BLUE for Leftward direction (degree of (135, 180] & (-180, -135])
                    else if ((dir[x, y, cur_frame] >= 135 && dir[x, y, cur_frame] <= 180) || (dir[x, y, cur_frame] >= -180 && dir[x, y, cur_frame] <= -135))
                    {
                        msm[x, y, 0] = value; //B
                        msm[x, y, 1] = 0; //G
                        msm[x, y, 2] = 0; //R
                    }
                    //YELLOW for Upward direction (degree of (-135, -45])
                    else if (dir[x, y, cur_frame] > -135 && dir[x, y, cur_frame] < -45)
                    {
                        msm[x, y, 0] = 0; //B
                        msm[x, y, 1] = value; //G
                        msm[x, y, 2] = value; //R
                    }
                }
            }
        }
        #endregion

        #region NEURAL NETWORK PROCESSING
        /// <summary>
        /// Drosophila motion vision neural network model processing
        /// </summary>
        /// <param name="img1">first input grayscale image in time</param>
        /// <param name="img2">second input grayscale image in time</param>
        /// <param name="t">timestamp</param>
        public void Drosophila_Motion_Vision_Processing(byte[,,] img1, byte[,,] img2, int t)
        {
            // Init and Reset
            int cur_frame = t % 2;
            int pre_frame = (t - 1) % 2;
            wfm_on_on_hor = 0;
            wfm_off_off_hor = 0;
            wfm_on_on_ver = 0;
            wfm_off_off_ver = 0;
            maxMag = 0;
            // Computational Retina layer processing
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    HPF[y, x, cur_frame] = HighpassFilter(img1[y, x, 0], img2[y, x, 0]);
                }
            }
            // Computational Lamina layer processing
            int[,] tmp_dog = DOG(HPF, cur_frame);
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    // Half-Wave Rectifier
                    HalfWaveRectifier(tmp_dog[y, x], y, x, cur_frame);
                    // FDSR mechanism for each polarity cell
                    FDSR(ONs[y, x, pre_frame], ref ONs[y, x, cur_frame], OFFs[y, x, pre_frame], ref OFFs[y, x, cur_frame]);
                }
            }
            // Computational Medulla and Lobula layers processing
            // Downstream processing
            // Symmetric structure of each pair-wise cells
            // Put bias in all inhibitory flows to get stronger responses to preferred direction
            // Dynamic low-pass filers
            for (int y = 0; y < height - Ncon * Ncon_step + Ncon_step; y++)
            {
                for (int x = 0; x < width - Ncon * Ncon_step + Ncon_step; x++)
                {
                    // Reset
                    ON_ON_hor[y, x, cur_frame] = 0;
                    ON_ON_ver[y, x, cur_frame] = 0;
                    OFF_OFF_hor[y, x, cur_frame] = 0;
                    OFF_OFF_ver[y, x, cur_frame] = 0;
                    emd_h[y, x, cur_frame] = 0;
                    emd_v[y, x, cur_frame] = 0;
                    for (int i = Ncon_step; i <= Ncon * Ncon_step - Ncon_step; i = i + Ncon_step)
                    {
                        // The dynamic first-order low-pass filtering for each pair of cells with dynamic time parameter
                        ON_lpf[y, x] = LowpassFilter(ONs[y, x, cur_frame], ONs[y, x, pre_frame], dyn_lp_delay[(i - Ncon_step) / Ncon_step]);
                        OFF_lpf[y, x] = LowpassFilter(OFFs[y, x, cur_frame], OFFs[y, x, pre_frame], dyn_lp_delay[(i - Ncon_step) / Ncon_step]);
                        ON_lpf[y, x + i] = LowpassFilter(ONs[y, x + i, cur_frame], ONs[y, x + i, pre_frame], dyn_lp_delay[(i - Ncon_step) / Ncon_step]);
                        ON_lpf[y + i, x] = LowpassFilter(ONs[y + i, x, cur_frame], ONs[y + i, x, pre_frame], dyn_lp_delay[(i - Ncon_step) / Ncon_step]);
                        OFF_lpf[y, x + i] = LowpassFilter(OFFs[y, x + i, cur_frame], OFFs[y, x + i, pre_frame], dyn_lp_delay[(i - Ncon_step) / Ncon_step]);
                        OFF_lpf[y + i, x] = LowpassFilter(OFFs[y + i, x, cur_frame], OFFs[y + i, x, pre_frame], dyn_lp_delay[(i - Ncon_step) / Ncon_step]);

                        // HORIZONTAL SYSTEM
                        // L1 Pathway
                        mlp1 = ON_lpf[y, x] * ONs[y, x + i, cur_frame];
                        mlp2 = ONs[y, x, cur_frame] * ON_lpf[y, x + i];
                        on_on_sub = mlp1 - inh_bias * mlp2;
                        // L2 Pathway
                        mlp1 = OFF_lpf[y, x] * OFFs[y, x + i, cur_frame];
                        mlp2 = OFFs[y, x, cur_frame] * OFF_lpf[y, x + i];
                        off_off_sub = mlp1 - inh_bias * mlp2;
                        // Horizontal detectors
                        emd_h[y, x, cur_frame] += (coe_a * on_on_sub + coe_b * off_off_sub + coe_c * on_on_sub * off_off_sub);
                        ON_ON_hor[y, x, cur_frame] += on_on_sub;
                        OFF_OFF_hor[y, x, cur_frame] += off_off_sub;

                        // VERTICAL SYSTEM
                        // L1 Pathway
                        mlp1 = ON_lpf[y, x] * ONs[y + i, x, cur_frame];
                        mlp2 = ONs[y, x, cur_frame] * ON_lpf[y + i, x];
                        on_on_sub = mlp1 - inh_bias * mlp2;
                        // L2 Pathway
                        mlp1 = OFF_lpf[y, x] * OFFs[y + i, x, cur_frame];
                        mlp2 = OFFs[y, x, cur_frame] * OFF_lpf[y + i, x];
                        off_off_sub = mlp1 - inh_bias * mlp2;
                        // Vertical detectors
                        emd_v[y, x, cur_frame] += (coe_a * on_on_sub + coe_b * off_off_sub + coe_c * on_on_sub * off_off_sub);
                        ON_ON_ver[y, x, cur_frame] += on_on_sub;
                        OFF_OFF_ver[y, x, cur_frame] += off_off_sub;
                    }
                    // Low-pass filter for T4 and T5 cells to reduce noise
                    ON_ON_hor[y, x, cur_frame] = LowpassFilter(ON_ON_hor[y, x, cur_frame], ON_ON_hor[y, x, pre_frame], emd_t);
                    ON_ON_ver[y, x, cur_frame] = LowpassFilter(ON_ON_ver[y, x, cur_frame], ON_ON_ver[y, x, pre_frame], emd_t);
                    OFF_OFF_hor[y, x, cur_frame] = LowpassFilter(OFF_OFF_hor[y, x, cur_frame], OFF_OFF_hor[y, x, pre_frame], emd_t);
                    OFF_OFF_ver[y, x, cur_frame] = LowpassFilter(OFF_OFF_ver[y, x, cur_frame], OFF_OFF_ver[y, x, pre_frame], emd_t);
                    // Wide field motion detectors -> T4, T5 cells integrated in LPTCs, directionally
                    wfm_on_on_hor += ON_ON_hor[y, x, cur_frame];
                    wfm_on_on_ver += ON_ON_ver[y, x, cur_frame];
                    wfm_off_off_hor += OFF_OFF_hor[y, x, cur_frame];
                    wfm_off_off_ver += OFF_OFF_ver[y, x, cur_frame];
                    // Motion magnitude
                    mag[y, x, cur_frame] = (float)(Math.Sqrt(Math.Pow(emd_h[y, x, cur_frame], 2) + Math.Pow(emd_v[y, x, cur_frame], 2)));
                    if (maxMag < mag[y, x, cur_frame])
                        maxMag = mag[y, x, cur_frame];
                    // Calculate motion direction in radians (atan2(y, x) * 180 / pi)
                    dir[y, x, cur_frame] = (float)(Math.Atan2(emd_v[y, x, cur_frame], emd_h[y, x, cur_frame]) * 180 / Math.PI);
                }
            }
            // Computational Lobula Plate layer processing
            // Activation after spatial integration of local T4/T5 signals
            if (wfm_on_on_hor >= 0)
                wfm_on_on_hor = SigmoidTransfer(wfm_on_on_hor) - delta_C;
            else
                wfm_on_on_hor = -SigmoidTransfer(Math.Abs(wfm_on_on_hor)) + delta_C;
            if (wfm_on_on_ver >= 0)
                wfm_on_on_ver = SigmoidTransfer(wfm_on_on_ver) - delta_C;
            else
                wfm_on_on_ver = -SigmoidTransfer(Math.Abs(wfm_on_on_ver)) + delta_C;
            if (wfm_off_off_hor >= 0)
                wfm_off_off_hor = SigmoidTransfer(wfm_off_off_hor) - delta_C;
            else
                wfm_off_off_hor = -SigmoidTransfer(Math.Abs(wfm_off_off_hor)) + delta_C;
            if (wfm_off_off_ver >= 0)
                wfm_off_off_ver = SigmoidTransfer(wfm_off_off_ver) - delta_C;
            else
                wfm_off_off_ver = -SigmoidTransfer(Math.Abs(wfm_off_off_ver)) + delta_C;

            // Horizontal and vertical membrane potential in Lobula Plate
            wfm_h = wfm_on_on_hor + wfm_off_off_hor;
            wfm_v = wfm_off_off_ver + wfm_on_on_ver;

            // Motion signal map matrix that can be visualised using OpenCV library: color indicates motion direction; proportional intensity indicates motion magnitude in the receptive field.
            MotionSM(t);

            Console.WriteLine("Frame {0} Wide Field Motion---Horizontal: {1:F} ---Vertical: {2:F}", t, wfm_h, wfm_v);
            Console.WriteLine("Frame {0} L1 WFM--------------Horizontal: {1:F} ---Vertical: {2:F}", t, wfm_on_on_hor, wfm_on_on_ver);
            Console.WriteLine("Frame {0} L2 WFM--------------Horizontal: {1:F} ---Vertical: {2:F}", t, wfm_off_off_hor, wfm_off_off_ver);
            Console.WriteLine("Frame {0} VO_H: {1:F} --- VO_V: {2:F}", t, vo_h, vo_v);

            //Visual Odometer accumulators in horizontal and vertical directions
            vo_h += wfm_h;
            vo_v += wfm_v;
        }
        #endregion
    }
}

