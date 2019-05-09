def wahio(n):
    lista=list(str(n))
    inverse,new,con,=lista[::-1],['','','','','','','','',''],0
    for i in inverse:
        new[con]=int(i)
        con+=1
    a,b,c,d,e,f,g,h,i=new[::-1]
    if len(str(new[3]))>0:
        new.insert(3,'.')
    if len(str(new[7]))>0:
        new.insert(7,'.')
    numero=new[::-1]
    print('\n'+'Resultado para: ',end='')
    for i in numero:
        print(str(i),end='')
    print()    
    unidad={1:'un', 2:'dos', 3:'tres', 4:'cuatro', 5:'cinco', 6:'seis', 7:'siete', 8:'ocho', 9:'nueve',0:'','':''}
    unidadi={1:'uno', 2:'dos', 3:'tres', 4:'cuatro', 5:'cinco', 6:'seis', 7:'siete', 8:'ocho', 9:'nueve',0:'','':''}
    unidad2={10:'diez', 11:'once', 12:'doce', 13:'trece', 14:'catorce', 15:'quince',16:'diez y seis',17:'diez y siete', 18:'diez y ocho', 19:'diez y nueve'}
    decena={1:'diez', 2:'veinti', 3:'treinta', 4:'cuarenta', 5:'cincuenta', 6:'sesenta', 7:'setenta', 8:'ochenta', 9:'noventa','':'',0:''}
    centena={1:'ciento', 2:'dos cientos',3:'tres cientos',4:'cuatro cientos',5:'quinientos',6:'seis cientos',7:'setecientos',8:'ocho cientos',9:'novecientos','':'',0:''}
 
    a=centena[a]
    if b==1 and c<6:
        b,c=unidad2[int(str(b)+str(c))],'millones'
    elif c==1:
        c,b='un millon',decena[b]
    elif b==0:
        b,c='',(unidad[c]+len(str(c))*' millones')
    else:
        b=(decena[b]+len(str(b))*' y')
        c=(unidad[c]+len(str(c))*' millones')
    d=centena[d]
    if e==1 and f<6:
        e,f=unidad2[int(str(e)+str(f))],'mil'
    elif f==0:
        e,f=decena[e],'mil'
    elif e==0:
        e,f='',(unidad[f]+len(str(f))*' mil')
    else:
        e=(decena[e]+len(str(e))*' y')
        f=(unidad[f]+len(str(f))*' mil')
    g=centena[g]
    if h==1 and i<6:
        h,i=unidad2[int(str(h)+str(i))],''
    elif h==0:
        h,i='',unidadi[i]
    else:
        if i==0:
            i,h='',decena[h]
        else:
            i,h=unidadi[i],decena[h]+len(str(h))*' y'
    orden=[a,b,c,d,e,f,g,h,i]
    return orden


if __name__ == '__main__':
	n=int(input())
	aa = wahio(n)
	cadena = ' '.join(aa)
	print(cadena.strip())