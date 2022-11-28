import time
import random as rnd
import tkinter as Tk
from learning import *
#a,b = b,a
network = Network(4,4)

def degistir(numara):
	global tiklama
	global ilk
	tiklama += 1 # tiklama = tiklama + 1
	if tiklama == 2:
		ikinci = numara
		#print(ilk,ikinci)
		ilksat, ilksut, ikincisat, ikincisut = ilk // sut, ilk % sut, ikinci // sut, ikinci % sut
		if abs(ilksat - ikincisat) + abs(ilksut -ikincisut) == 1: 
			if ilk<ikinci:
					if network.feedforward([int(dugmeler[ilk]["text"]), int(dugmeler[ikinci]["text"])])<0.5:
						dugmeler[ilk]["text"], dugmeler[ikinci]["text"] = dugmeler[ikinci]["text"], dugmeler[ilk]["text"]
						#print("Değiş")
						tiklama = 0
					else:
						#print("Değişme")
						ilk = numara
			
		tiklama = 0
	else:
		ilk = numara

tiklama = 0
renk = ["Yellow", "Red"]


def Buton():
	
	for y in range(10000):
		a=rnd.randint(0,99)
		b=rnd.randint(0,99)
		
		degistir(a)
		degistir(b)
		

sat, sut = 10,10
form = Tk.Tk()
a,b,c,d = 700, 600, (form.winfo_screenwidth()-600)//2, (form.winfo_screenheight()-400)//2
form.title("İlk Formumuz")
form.geometry("%sx%s+%s+%s"%(a,b,c,d))
b1=Tk.Button(form, text ="Çözümle", command = Buton)
dugmeler = []
b1.pack()
secim = 0
yuk, gen = 45, 45
liste = rnd.sample(range(1,sat*sut+1), sat*sut)
for row in range(sat):
	for column in range(sut):
		dugmeler.append(Tk.Button(text = str(liste[row*sut+column]), bg = renk[rnd.randint(0,secim)], command = lambda x = row*sut+column : degistir(x) ))
		dugmeler[row*sut+column].place(x = column*gen+20, y=yuk*row+20,height = yuk, width = gen)
Tk.mainloop()