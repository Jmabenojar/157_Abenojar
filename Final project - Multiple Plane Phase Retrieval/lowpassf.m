function [u_out,fu_out] = lowpassf(u_in,r)
[M,N] = size(u_in);
x = -M/2:M/2-1;
y = x;
[X,Y] = meshgrid(x,y);
filter = X.^2+Y.^2 < r^2;
fu_in = fftshift(fft2(u_in));
fu_out = fu_in.*filter;
% imwrite(uint8(normalize(crp(abs(fu_out),[800/2 800/2],208))),['C:\Users\Admin\OneDrive - University of the Philippines\MATLAB\THESIS IMAGE DUMP','\Spec.bmp']);
u_out = ifft2(ifftshift(fu_out));
end