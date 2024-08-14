function[u2]=prop(u1,lambda,z,dx,method);
% propagation - impulse response approach
% assumes same x and y side lengths and
% uniform sampling
% u1 - source plane field
% L - source and observation plane side length
% lambda - wavelength
% z - propagation distance
% u2 - observation plane field
% method - 'AS','TF','IR'
[M,N]=size(u1); %get input field array size
L=dx*M; %sample interval
k=2*pi/lambda; %wavenumber

fx=-1/(2*dx):1/L:1/(2*dx)-1/L; %freq coords
[FX,FY]=meshgrid(fx,fx);

x=-L/2:dx:L/2-dx; %spatial coords
[X,Y]=meshgrid(x,x);

U1=fft2(fftshift(u1)); %shift, fft src field


switch upper(method)
    case 'TF'
        H=fftshift(exp(1i*pi*lambda*z*(FX.^2+FY.^2)));
    case 'AS'
        H=fftshift(exp(-1i*k*z.*sqrt(1-(lambda.*FX).^2-(lambda.*FY).^2)));
    case 'AS2'
        H=fftshift(exp(-1i*k*z.*sqrt(1-(lambda.*FX).^2-(lambda.*FY).^2)));
        H(sqrt(FX.^2+FY.^2)>=1/lambda)=0;
    case 'IR'
        h=1/(1i*lambda*z)*exp(-1i*k/(2*z)*(X.^2+Y.^2)); %impulse
        H=fft2(fftshift(h))*dx^2;
    otherwise
        error('Unknown propagation method. Select "TF,AS, or IR" ')
end
U2=H.*U1; %multiply
u2=ifftshift(ifft2(U2)); %inv fft, center obs field
end