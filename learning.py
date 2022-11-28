import numpy as np

## Yapay Sinir Ağı Python Class'ı (Python Sınıfı)

"""
#Eğer katsayılar ve hata miktarındaki değişimi grafik olarak görmek isterseniz bu blok açık kalacak

hatalar = []
a1 = []
a2 = []
a3 = []
a4 = []
a5 = []
a6 = []
bi1 = []
bi2 = []
bi3 = []
"""
class Network:     
    
    ## Ağırlıklarımızı ve bias değerlerimizi burada oluşturulmaktadır.
    def __init__(self,sat, sut):
        self.sat = sat
        self.sut = sut

        # Ağ üzerinden 3 adet nöron olduğu için 
        # 6 adet ağırlık ve 3 adet bias değeri olmalı

        
        # Bu blok eğitim yapılırken açılacak onun dışında kapalı kalacak
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()
        
        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()
        
        # Alttaki satır eğitim yapılırken kapalı olacak
        self.w1,self.w2,self.w3,self.w4,self.w5,self.w6,self.b1,self.b2,self.b3 = 4.475262695989136 , -4.4677068739156995 , -2.8782856635576364 , 2.9264097876329194 , -8.91748734029233 , 5.082832417967009 , -2.2137284054096726 , 1.0690239833121857 , 2.0927625746737677    ## Sigmoid fonksiyonu 
    def sigmoid(self , x): 
        
        # Sigmoid aktivasyon fonksiyonu : f(x) = 1 / (1 + e^(-x))
        return 1 / (1 + np.exp(-x))
    
    
    ## Sigmoid fonksiyonunun türevi
    def sigmoid_turev(self , x):
        
        # Sigmoid fonksiyonunun türevi: f'(x) = f(x) * (1 - f(x))
        sig    = self.sigmoid(x)
        result = sig * (1 - sig)        
        return result
            
    def mse_loss(self , y_real , y_prediction):
        
        # y_real ve y_prediction aynı boyutta numpy arrayleri olmalıdır. 
        return ((y_real - y_prediction) ** 2).mean()
    
    ## İleri beslemeli nöronlar üzerinden tahmin
    ## değerinin elde edilmesi 
    
    def feedforward(self , row):
        
        # h1 nöronunun değeri
        h1 = self.sigmoid((self.w1 * row[0]) + (self.w2 * row[1]) + self.b1 )
        
        # h2 nöronunun değeri
        h2 = self.sigmoid((self.w3 * row[0]) + (self.w4 * row[1]) + self.b2 )
        
        # Tahmin değeri 01 nöronun değeri
        o1 = self.sigmoid((self.w5 * h1 ) + (self.w6 * h2 ) + self.b3 )
        
        return o1         
    
    ## Belitiler iteresyon sayısı kadar model eğitimi
    def train(self , data , labels):
    
        learning_rate = 0.001
        epochs = 1000 # Eğitimde uygulanacak tekrar sayısı
        
        for epoch in range(epochs):
            
            for x, y in zip(data , labels):
                    
                # Neuron H1
                sumH1 = (self.w1 * x[0]) + (self.w2 * x[1]) + self.b1 
                H1    = self.sigmoid(sumH1)
                
                # Neuron H2
                sumH2 = (self.w3 * x[0]) + (self.w4 * x[1]) + self.b2
                H2    = self.sigmoid(sumH2)
                
                # Neuron O1
                sumO1 = (self.w5 * H1) + (self.w6 * H2) + self.b3
                O1    = self.sigmoid(sumO1)
                
                # Tahmin değerimiz
                prediction = O1
                
                # Türevlerin Hesaplanması 
                # dL/dYpred :  y = doğru değer | prediciton: tahmin değeri
                dLoss_dPrediction = -2*(y - prediction)
                
                # Nöron H1 için ağırlık ve bias türevleri 
                dH1_dW1 = x[0] * self.sigmoid_turev(sumH1)
                dH1_dW2 = x[1] * self.sigmoid_turev(sumH1)
                dH1_dB1 = self.sigmoid_turev(sumH1)
                
                
                # Nöron H2 için ağırlık ve bias türevleri
                dH2_dW3 = x[0] * self.sigmoid_turev(sumH2)
                dH2_dW4 = x[1] * self.sigmoid_turev(sumH2)
                dH2_dB2 = self.sigmoid_turev(sumH2)
                
                # Nöron O1 (output) için ağırlık ve bias türevleri
                dPrediction_dW5 = H1 * self.sigmoid_turev(sumO1) 
                dPrediction_dW6 = H2 * self.sigmoid_turev(sumO1) 
                dPrediction_dB3 = self.sigmoid_turev(sumO1) 
                
                # Aynı zamanda tahmin değerinin H1 ve H2'ye göre türevlerinin de
                # hesaplanması gerekmektedir. 
                dPrediction_dH1 = self.w5 * self.sigmoid_turev(sumO1)
                dPrediction_dH2 = self.w6 * self.sigmoid_turev(sumO1)
                
                ## Ağırlık ve biasların güncellenmesi 
                
                # H1 nöronu için güncelleme
                self.w1 = self.w1 - (learning_rate * dLoss_dPrediction * dPrediction_dH1 * dH1_dW1)
                self.w2 = self.w2 - (learning_rate * dLoss_dPrediction * dPrediction_dH1 * dH1_dW2)
                self.b1 = self.b1 - (learning_rate * dLoss_dPrediction * dPrediction_dH1 * dH1_dB1)
                
                # H2 nöronu için güncelleme 
                self.w3 = self.w3 - (learning_rate * dLoss_dPrediction * dPrediction_dH2 * dH2_dW3)
                self.w4 = self.w4 - (learning_rate * dLoss_dPrediction * dPrediction_dH2 * dH2_dW4)
                self.b2 = self.b2 - (learning_rate * dLoss_dPrediction * dPrediction_dH2 * dH2_dB2)
                
                # O1 nöronu için güncelleme 
                self.w5 = self.w5 - (learning_rate * dLoss_dPrediction * dPrediction_dW5) 
                self.w6 = self.w6 - (learning_rate * dLoss_dPrediction * dPrediction_dW6) 
                self.b3 = self.b3 - (learning_rate * dLoss_dPrediction * dPrediction_dB3) 
                
            predictions = np.apply_along_axis(self.feedforward ,1, data)
            loss = self.mse_loss(labels , predictions)

            """
            # Eğer katsayılar ve hata miktarındaki değişimi grafik olarak görmek isterseniz bu blok açık kalacak
            hatalar.append(loss)
            a1.append(self.w1)
            a2.append(self.w2)
            a3.append(self.w3)
            a4.append(self.w4)
            a5.append(self.w5)
            a6.append(self.w6)
            bi1.append(self.b1)
            bi2.append(self.b2)
            bi3.append(self.b3)
            """
            print("Devir %d loss: %.7f" % (epoch, loss))
            

        print(self.w1,",",self.w2,",",self.w3,",",self.w4,",",self.w5,",",self.w6,",",self.b1,",",self.b2,",",self.b3)
    
    def test(self):
        data=[]
        for x in range(self.sat):
            for y in range(self.sut):
                data.append([x,y])  

        labels=[]
        for x in range(len(data)):

            if data[x][0]>data[x][1]:
                labels.append(0)
            else:
                labels.append(1)
        
        return (np.array(data), np.array(labels))


    def testverisi(self):
        """ Bu metod hazırlayacağınız test verisi için size örnek olarak verilmiştir
        Kendi uygulamanız için uygun test verisini hazırlarken bundan faydalanabilirsiniz
        data = []
        labels = []
        for kdyeri in list(range(self.sat*self.sut)):
            ddliste = list(range(self.sat*self.sut))
            ddliste.pop(kdyeri)
            for ddyeri in ddliste:
                ilksat, ilksut, ikincisat, ikincisut = kdyeri // self.sut, kdyeri % self.sut, ddyeri // self.sut, ddyeri % self.sut
                data.append([kdyeri,ddyeri])
                labels.append(int((abs(ilksat - ikincisat) + abs(ilksut -ikincisut) == 1)))
        
        return (np.array(data), np.array(labels)) """


# Bu blok eğitim yapılırken açılacak onun dışında kapalı kalacak
"""
network = Network(10,10)
a = network.test()

network.train(*a)   
"""
# Donen veriyi kullan
network = Network(4,4)
a = network.feedforward([5,4])


"""
#Eğer katsayılar ve hata miktarındaki değişimi grafik olarak görmek isterseniz bu blok açık kalacak

import matplotlib.pyplot as plt

plt.plot(hatalar)
plt.plot(a1)
plt.plot(a2)
plt.plot(a3)
plt.plot(a4)
plt.plot(a5)
plt.plot(a6)
plt.plot(bi1)
plt.plot(bi2)
plt.plot(bi3)
plt.show()
"""

