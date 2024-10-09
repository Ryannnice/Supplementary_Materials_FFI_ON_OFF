import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import time 

class LGMD2:
    def LGMD(self):
        start_time = time.time()  # start time
        self.K = [] # membrane potential
        self.K.append(0.5)
        self.Kf = []# membrane potential before the SFA
        self.w1 = 0.0  #Son's Ion coefficient  Eon(x,y,t) - w1(t) * Ion(x,y,t) 
        self.w2 = 0.0  #Sff's Iff coefficient  Eff(x,y,t) - w2(t) * Iff(x,y,t)
        self.w3 = 1.0  # bias baseline in ON channels : max(w3, PM^(t)/Tpm)
        self.w4 = 0.3  # bias baseline in OFF channels : max(w4, PM^(t)/Tpm) 0.5 0.3 
        self.Tpm = 6.0  # threshold in PM way 10 8 
        self.a1 = 0.1 # coefficient in half-wave rectifying
        self.t = 53.0  # t is the time interval between the consecutive frames of digital signs(ms)  30~60
        self.t1 = 45.0  # delay time of the on pathway
        self.t2 = 30.0  # delay time of the on pathway
        self.t3 = 15.0  # delay time of the on pathway
        self.t4 = 180.0  # delay time of the off pathway
        self.t5 = 120.0  # delay time of the off pathway
        self.t6 = 60.0  # delay time of the off pathway
        self.a4 = self.t / (10 + self.t)  # PM^(t) = a4*PM(t) +(1-a4)*PM(t-1) 90 
        self.a5 = 0.6 # LGMD2 Cell  0.5~1  0.5 1.0 
        self.a6 = 800 / (800 + self.t)  # time constant in SFA  500~1000  500 800  
        self.pon0 = 0.0  # p0 represents PM(t-1)
        self.pff0 = 0.0  # p0 represents PM(t-1)
        self.p0 = 0.0  # p0 represents PM(t-1)
        self.o1 = 1.0  #S(x,y,t) = o1 * Son_cur(x,y,t) + o2 * Sff_cur(x,y,t) + o3 * Son_cur(x,y,t) * Sff_cur(x,y,t) 0.5 1,0 0.7
        self.o2 = 1.0  #S(x,y,t) = o1 * Son_cur(x,y,t) + o2 * Sff_cur(x,y,t) + o3 * Son_cur(x,y,t) * Sff_cur(x,y,t)
        self.o3 = 0.0  #S(x,y,t) = o1 * Son_cur(x,y,t) + o2 * Sff_cur(x,y,t) + o3 * Son_cur(x,y,t) * Sff_cur(x,y,t) 1.0 0.0 0.3 
        self.oon = 1.15   # threshold in the on pathway's close
        self.ooff = 1.15  # threshold in the off pathway's close
        self.Cde = 0.5
        self.Tde = 15.0
        self.Tsfa = 0.002  # small threshold in SFA 0.003 0.001 

        self.catch = cv2.VideoCapture(PathVal)
        self.height = int(self.catch.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width = int(self.catch.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.framescount = int(self.catch.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = int(self.catch.get(cv2.CAP_PROP_FPS))
        #self.t = 1000 / self.fps  # ms 
        self.n = self.height * self.width  # the number of cells
        self.AIM,S_spike,self.Spike,self.Col = [0] * self.framescount,[0] * self.framescount,[0] * self.framescount,[0] * self.framescount # feed forward inhibition


        a21 = self.t / (self.t + self.t1)  #40
        a22 = self.t / (self.t + self.t2)  #30
        a23 = self.t / (self.t + self.t3)  #15

        a31 = self.t / (self.t + self.t4)  # 180
        a32 = self.t / (self.t + self.t5)  # 120
        a33 = self.t / (self.t + self.t6)  # 60

        a2 = np.array([[a21, a22, a21], [a22, a23, a22], [a21, a22, a21]])# the kernel of temporal convolution (ON pathway)
        a2_1 = 1 - a2
        a3 = np.array([[a31, a32, a31], [a32, a33, a32], [a31, a32, a31]])# the kernel of temporal convolution (OFF pathway)
        a3_1 = 1 - a3

        W1_kernel = np.array([[0.25, 0.5, 0.25], [0.5, 2, 0.5], [0.25, 0.5, 0.25]])  # ON/OFF way's the kernel to gain Ion
        W2_kernel = np.array([[0.125, 0.25, 0.125], [0.25, 1, 0.25], [0.125, 0.25, 0.125]])  # ON/OFF way's the kernel to gain Iff
        Wg_kernel = np.array([[1.0 / 9, 1.0 / 9, 1.0 / 9], [1.0 / 9, 1.0 / 9, 1.0 / 9], [1.0 / 9, 1.0 / 9, 1.0 / 9]])  # G layer's the kernel to gain coefficient Matrix Ce(x,y,t)


        k1 = 0
        k_pre = 0
        Z=[]

        for t in range(0, self.framescount, 1):  # Skip frame or Reduce image dimension
            ret, frame = self.catch.read()
            X = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            Y = X.astype(np.float32)
            Z.append(Y)
            #cv2.imwrite(r"C:\Users\21124\Desktop\pre_output\pre_gray_{}.png".format(t), Z[:, :, t])

        P,Pon,Pff,Eon,Eff,Son,Sff= np.zeros((self.height, self.width, len(Z)), dtype=np.float32),np.zeros((self.height, self.width, len(Z)), dtype=np.float32),np.zeros((self.height, self.width, len(Z)), dtype=np.float32),np.zeros((self.height, self.width, len(Z)), dtype=np.float32),np.zeros((self.height, self.width, len(Z)), dtype=np.float32),np.zeros((self.height, self.width, len(Z)), dtype=np.float32),np.zeros((self.height, self.width, len(Z)), dtype=np.float32) 

        for t in range(2, len(Z)): 
            a11 = 1/(1+np.exp(t-1))
            a12 = 1/(1+np.exp(t-2))
            a = 1/(1+np.exp(0))
            P[:, :, t] = Z[t] - Z[t-1] + a11*P[:, :, t-1] + a12*P[:, :, t-2] 
            #cv2.imwrite(r"C:\Users\21124\Desktop\P_output\P_gray_{}.png".format(t), P[:, :, t])
    
            # half_wave rectify 
            c0 = np.greater(P[:, :, t], 0).astype(int) 
            Pon[:, :, t] = np.multiply(P[:, :, t], c0) + 0.1 * Pon[:, :, t-1] 
            c1 = np.less(P[:, :, t], 0).astype(int) 
            Pff[:, :, t] = np.abs(np.multiply(P[:, :, t], c1)) + 0.1 * Pff[:, :, t-1]
            # cv2.imwrite(r"C:\Users\21124\Desktop\P_output\on_output\on_gray_{}.png".format(i), Pon_cur)  
            # cv2.imwrite(r"C:\Users\21124\Desktop\P_output\off_output\off_gray_{}.png“.format(i), Pff_cur)

            # Adaptive Inhibition Mechanism
            s1 = 0
            for row in range(self.height):
                for col in range(self.width):
                    s1 = s1 + np.abs(Pon[row,col,t])
            s2 = 0
            for row in range(self.height):
                for col in range(self.width):
                    s2 = s2 + np.abs(Pff[row,col,t])
            pmon = s1 / self.n  # pmon is ON way's PM(t)
            pmff= s2 / self.n  # pmoff is OFF way's PM(t)
            PM_on = self.a4 * pmon + (1 - self.a4) * self.pon0  # ON way PM^(t) = a4*PM(t) +(1-a4)*PM(t-1)
            PM_ff = self.a4 * pmon + (1 - self.a4) * self.pff0  # OFF way PM^(t) = a4*PM(t) +(1-a4)*PM(t-1)
            
            won0 = PM_on / self.Tpm
            wff0 = PM_ff / self.Tpm

            # calculate w1(t) = max(w3, PM^(t)/Tpm)
            if won0 > self.w3:
                self.w1 = won0 
            else:
                self.w1 = self.w3

            # calculate w2(t) = max(w4, PM^(t)/Tpm)
            if wff0 > self.w4:
                self.w2 = wff0
            else:
                self.w2 = self.w4

            self.AIM[t] = (PM_on+PM_ff)

            self.pon0 = pmon
            self.pff0 = pmff

            if PM_ff >= self.ooff :
                Eff[:,:,t] = Pff[:,:,t]
                for x in range(1, self.height-1):
                    for y in range(1, self.width-1):
                        center2 = (Eff[x][y][t] * a3[1][1] + Eff[x][y][t-1] * a3_1[1][1]) * W2_kernel[1][1]
                        diagonal2 = ((Eff[x - 1][y - 1][t] * a3[0][0] + Eff[x - 1][y - 1][t-1] * a3_1[0][0]) * W2_kernel[0][0]
                                    + (Eff[x - 1][y + 1][t] * a3[0][2] + Eff[x - 1][y + 1][t-1] * a3_1[0][2]) * W2_kernel[0][2]
                                    + (Eff[x + 1][y - 1][t] * a3[2][0] + Eff[x + 1][y - 1][t-1] * a3_1[2][0]) * W2_kernel[2][0]
                                    + (Eff[x + 1][y + 1][t] * a3[2][2] + Eff[x + 1][y + 1][t-1] * a3_1[2][2]) * W2_kernel[2][2])
                        adjacent2 = ((Eff[x - 1][y][t] * a3[0][1] + Eff[x - 1][y][t-1] * a3_1[0][1]) * W2_kernel[0][1]
                                    + (Eff[x][y + 1][t] * a3[1][2] + Eff[x][y + 1][t-1] * a3_1[1][2]) * W2_kernel[1][2]
                                    + (Eff[x][y - 1][t] * a3[1][0] + Eff[x][y - 1][t-1] * a3_1[1][0]) * W2_kernel[1][0]
                                    + (Eff[x + 1][y][t] * a3[2][1] + Eff[x + 1][y][t-1] * a3_1[2][1]) * W2_kernel[2][1])
                        Iff = center2 + diagonal2 + adjacent2

                        Sff[x][y][t] = Eff[x][y][t] - Iff* self.w2
                        if Sff[x][y][t] < 0:
                            Sff[x][y][t] = 0
            else:
                Sff[:,:,t]=np.zeros((self.height, self.width))

            if PM_on >= self.oon :
                Eon[:,:,t] = Pon[:,:,t] 
                for x in range(1, self.height-1):
                    for y in range(1, self.width-1):
                        center1 = (Eon[x][y][t] * a2[1][1] + Eon[x][y][t-1] * a2_1[1][1]) * W1_kernel[1][1]
                        diagonal1 = ((Eon[x-1][y-1][t] * a2[0][0] + Eon[x-1][y-1][t-1] * a2_1[0][0]) * W1_kernel[0][0]
                                    + (Eon[x-1][y+1][t] * a2[0][2] + Eon[x-1][y+1][t-1] * a2_1[0][2]) * W1_kernel[0][2]
                                    + (Eon[x+1][y-1][t] * a2[2][0] + Eon[x+1][y-1][t-1] * a2_1[2][0]) * W1_kernel[2][0]
                                    + (Eon[x+1][y+1][t] * a2[2][2] + Eon[x+1][y+1][t-1] * a2_1[2][2]) * W1_kernel[2][2])
                        adjacent1 = ((Eon[x-1][y][t] * a2[0][1] + Eon[x-1][y][t-1] * a2_1[0][1]) * W1_kernel[0][1]
                                    + (Eon[x][y+1][t] * a2[1][2] + Eon[x][y+1][t-1] * a2_1[1][2]) * W1_kernel[1][2]
                                    + (Eon[x][y-1][t] * a2[1][0] + Eon[x][y-1][t-1] * a2_1[1][0]) * W1_kernel[1][0]
                                    + (Eon[x+1][y][t] * a2[2][1] + Eon[x+1][y][t-1] * a2_1[2][1]) * W1_kernel[2][1])
                        Ion = center1 + diagonal1 + adjacent1
                        Son[x][y][t] = Eon[x][y][t] - Ion * self.w1
                        if Son[x][y][t] < 0:
                            Son[x][y][t] = 0
            else:
                Son[:,:,t]=np.zeros((self.height, self.width))

            # gain S(x,y,t) = o1 * Son_cur(x,y,t) + o2 * Sff_cur(x,y,t) + o3 * Son_cur(x,y,t) * Sff_cur(x,y,t)
            S = self.o1 * Son[:,:,t] + self.o2 * Sff[:,:,t] + self.o3 * Son[:,:,t] * Sff[:,:,t]
            #cv2.imwrite(r"C:\Users\21124\Desktop\S_output\S_gray_{}.png".format(i), S[:,:,t])
            
            # gain G(x,y,t)
            Ce = cv2.filter2D(S, -1, Wg_kernel)
            w = np.max(Ce) / 4.0 + 0.01
            G = np.multiply(S, Ce) / w
            for x in range(self.height):
                for y in range(self.width):
                    if np.any(G[x,y] * self.Cde < self.Tde):
                        G[x,y] = 0
            #cv2.imwrite(r"C:\Users\21124\Desktop\G_output\G_gray_{}.png".format(i), G[:,:,t]) 

            # gain membrane potential
            mp = np.sum(G)
            k = 1.0 / (1 + math.exp(-mp / (self.n * self.a5)))
            self.Kf.append(k)
            if self.AIM[t] >= 10+0.02*self.AIM[t-1]:
                k = 0.5

            # SFA mechanism
            # K^(t) = a6 * (K^(t-1) + K(t) - K(t-1))
            # k represents K(t), k_pre represents K^(t-1), k_cur represents K^(t), k1 represents K(t-1)
            if k - k1 <= self.Tsfa:
                k_cur = self.a6 * (k_pre + k - k1)
                if k_cur < 0.5:
                    k_cur = 0.5
            else:
                k_cur = self.a6 * k
                if k_cur < 0.5:
                    k_cur = 0.5
            

            self.K.append(k_cur)
            k1 = k
            k_pre = k_cur

            self.a7 = 4
            T_spi = 0.78
            n_ts = 4
            n_sp = 3
            n_sp2 = 6

            S_spike[t] = np.exp(self.a7 * (k_cur - T_spi))
            if t < n_ts+2:
                self.Spike[3] = S_spike[3] + S_spike[2] 
                self.Spike[4] = S_spike[4] + S_spike[3] + S_spike[2] 
                self.Spike[5] = S_spike[5] + S_spike[4] + S_spike[3] + S_spike[2]
            else:
                self.Spike[t] = np.sum(S_spike[t - n_ts:t])

            if self.Spike[t] >= n_sp:
                self.Col[t-3] = 1
            else:
                self.Col[t-3] = 0

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Total runtime: {elapsed_time:.2f} seconds")



        curve_color = (0/255, 0/255, 234/255)
        dashed_line_color = (214/255, 226/255, 228/255)
        dashed_color = (164/255, 166/255, 168/255)

        x = list(range(len(self.K)))
        y = list(range(len(self.Col)))

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.axhline(y=0.78, color=dashed_color, linestyle='--',linewidth=2.5)
        ax.axhline(y=1, color=dashed_line_color)

        for i in range(len(y)):
            if i==0:
                ax.plot([y[i], y[i]], [0, self.Col[i]], color='r',linestyle='--', linewidth=2.3, label='Spike', alpha=0.6)
                ax.plot(y[i], self.Col[i], marker='o', markersize=6, color='r')  # 添加圆点
            else:
                ax.plot([y[i], y[i]], [0, self.Col[i]], color='r',linestyle='--', linewidth=2.3, alpha=0.6)
                ax.plot(y[i], self.Col[i], marker='o', markersize=6, color='r')  # 添加圆点

        ax.plot(x, self.K, color=curve_color, linewidth=2.7,label='oLGMD2')
        ax.set_ylim([0.45, 1.1])
        plt.yticks([0.5, 0.8, 1.0],fontsize=30)
        plt.xticks(fontsize=30) 
        ax.set_ylabel('Membrane Potential', fontsize=32)

        
        K_array = np.array(self.K)

        #ax.set_xlabel('frames', fontsize=16)
        ax.legend(loc='upper left',fontsize=32)

        plt.tight_layout()
        plt.show()


PathVal = r"C:\Users\21124\Desktop\LGMD\LGMD_video\discuss 0408\vehicle stimuli\c04.mp4"


if __name__ == '__main__':
    lgmd2 = LGMD2()
    lgmd2.LGMD() 

