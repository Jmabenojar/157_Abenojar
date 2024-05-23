function  I_out = crp(I_in,centr,arrysize)
I_out = I_in(centr(1)-floor(arrysize/2):centr(1)+floor(arrysize/2)-1,centr(2)-floor(arrysize/2):centr(2)+floor(arrysize/2)-1);
end