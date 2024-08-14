function [u_rec,tFB,mse_amp_fb,mse_ph_fb] = sbmir(lambda,cp,dist,z0,num,iter,amps,gph,root_to,method)
%%Conventional SBMIR Algorithm
%lambda - wavelength
%cp - camera pitch
%dist - distance between measurement planes
%num - number of measurement planes
%iter - number of iterations
%u - an array containing the amplitudes
%gph - guess phase
%u_rec - reconstruct wave field
%tFB - time of completion of sbmir
%mse_amp_fb - mse per iteration
%u_outs - reconstructed wavefields save in an array
%root_to - save to this path

phase = gph;
%% CONSTRUCT TRANSFER FUNCTIONS
[C,R] = size(amps(:,:,1)); %get aperture size
L = cp*C;  %side length
fx = -1/(2*cp):1/L:1/(2*cp)-1/L;
[FX,FY] = meshgrid(fx,fx);
k = 2*pi/lambda;

% Sf =  fftshift(exp(-1i*k*dist.*sqrt(1-(lambda.*FX).^2-(lambda.*FY).^2))); %forward propagate to next measurement plane
% Sb = fftshift(exp(1i*k*dist.*sqrt(1-(lambda.*FX).^2-(lambda.*FY).^2))); %back propagate to next measurement plane
% St = fftshift(exp(1i*k*z0.*sqrt(1-(lambda.*FX).^2-(lambda.*FY).^2))); %back propagate from 1st plane to object plane

%% MSE
mse_amp_fb = [0];
mse_ph_fb = [0];
%% SMOOTH FUNCTION
smth = @(ph,n) atan2(conv2(sin(ph),ones(n)/2,'same'),conv2(cos(ph),ones(n)/2,'same'));

%% ITERATION
tic
for iii =1:iter
    ph=phase;
    for ii = 1:num-1
        a = amps(:,:,ii);
        % forward= ifftshift(ifft2(fft2(fftshift()).*(Sf)));
        forward = prop(a.*exp(1i*phase),lambda,dist,cp,method);
        phase = angle(forward);
    end  
    for ii = 1:num-1
       a = amps(:,:,num+1-ii);
       % backward = ifftshift(ifft2(fft2(fftshift(a.*exp(1i*phase))).*(Sb)));
       backward = prop(a.*exp(1i*phase),lambda,-dist,cp,method);
       phase = angle(backward);
    end
    u_rec = amps(:,:,1).*exp(1i*phase);
    a2 = abs(backward);
    err_amp = mean(mean((a2-amps(:,:,1)).^2)); %amplitude error
    err_ph = mean(mean((ph-phase).^2)); %phase error
    mse_amp_fb = [mse_amp_fb;err_amp];
    mse_ph_fb = [mse_ph_fb;err_ph];
    % u_rec = ifftshift(ifft2(fft2(fftshift(u_rec)).*(St))); %back propagate to object plane
    u_rec=prop(u_rec,lambda,-z0,cp,method);
    if isreal(iii) && rem(iii,1)==0
%        imwrite(uint8(normalize(abs(u_rec))),[root_to,'\AmpFB_num=',num2str(num),'_iter=',num2str(iii),'.bmp']);
%        imwrite(uint8(normalize(smth(angle(u_rec),10))),[root_to,'\PhFB_num=',num2str(num),'_iter=',num2str(iii),'.bmp']);
       imwrite(uint8(normalize(abs(u_rec))),[root_to,'\Amp_num=',num2str(num),'_iter=',num2str(iii),'.bmp']);
       imwrite(uint8(normalize(smth(angle(u_rec),10))),[root_to,'\Ph_num=',num2str(num),'_iter=',num2str(iii),'.bmp']);
    end
end
tFB = toc;
end