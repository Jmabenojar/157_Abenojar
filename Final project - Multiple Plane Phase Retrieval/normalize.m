function [Iout]=normalize(Iin)
mn=min(min(Iin));
Iout=Iin-mn;
mx=max(max(Iout));
fact=255/mx;
Iout=fact.*Iout;
end