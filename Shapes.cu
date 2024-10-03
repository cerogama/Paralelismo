#include "bitmap_image.hpp"
#include <string>
#include <stdio.h>
using namespace std;

int main()
{
    //bitmap_image image(16, 16);
    bitmap_image cruz(16, 16);
    bitmap_image equis(16, 16);
    bitmap_image circulo(33, 33);

    int CenteroX = 16;
    int CenteroY = 16;
    int Radio = 16;



    //Hacemos la cruz dandole un punto de inicio con un ofset en y para que baje y en x para que se desplace hacia la derecha
    for (int i = 0; i < 16; i++)
    {
        cruz.set_pixel(i, 8, 211, 50, 218);//8 es el ofset en y
        cruz.set_pixel(i, 7, 211, 50, 218);//8 es el ofset en y
        cruz.set_pixel(8, i, 211, 50, 218);//8 es el ofset en x
        cruz.set_pixel(7, i, 211, 50, 218);//8 es el ofset en x
        //Se agrego las lineas 16 y 18 para darle simetria a la cruz
    }
    
    
  
    /*
    Los estoy guardando por bloques momentaneamente para que se guarde uno y despues se cree el otro
    */

    
    //hacemos la diagonal inicial que va de iteracion en iteracion y a eso le pintamos la misma diagonal
    //con yn ofset en y de la cantidad de pixeles menos uno para que pinte en una diagonal negativa
    for (int i= 0; i < 16; i++)
    {
        int c;
        
        equis.set_pixel(i, i, 211, 50, 218);
        equis.set_pixel(i, 15 - i, 211, 50, 218);
        //se le da el ofset en y, y se le resta la cantidad de iteraciones
        //y es lo que nos da la diagonal invertida
    }

    /*
    si queremos que el circulo nos quede perfecto las primeras variables generadas tienen que ser la mitas -1 del tamaño de la imagen 
    Ejemplo:
    si nuestra imagen mide 16 * 16 la mitad serian 8 y para que quede perfecto tendriamos que restarle 1 entonces,
    tendriamos que los primeros 3 valores tienen que ser 7 
    */



    for (int i = -Radio; i <= Radio; i++)
    {
        for (int j = -Radio; j <= Radio; j++)
        {
            if (j*j + i*i <= Radio * Radio)
            {
                int PixelI = CenteroX + i  ;//puede que esten alreves pero mira si sale quien soy yo para negarme
                int PixelJ = CenteroY + j  ;
                if (PixelI >= 0 && PixelI < 33 && PixelJ >= 0 && PixelJ < 33)//Estos son los limites de donde se pintan
                {
                    circulo.set_pixel(PixelI, PixelJ, 211, 50, 218);
                }
            }
        }
    }

    /*
        curiosamente con numeros pares me queda un desface de 1 pixel siempre, aunque se puede arreglar bajandole un pixel
        al tamaño de la imagen y a los limites de dibujo de esta no se encuentra en el formato estandar o al que globalmente se maneja,
        yo recomiendo bajarle un pixel a los limites de dibujo y al tamaño de la imagensi quieres el triangulo perfecto, si no pus no dejalo en estandar
        
        En Resumen: 
        la imagen del sirculo perfecto solo funciona con numeros nones en los lñimites de dibujo y tamaño de la imagen
    */
    cruz.save_image("cruz.bmp");
    equis.save_image("equis.bmp");
    circulo.save_image("circuclo.bmp");
    
    return 0;
}