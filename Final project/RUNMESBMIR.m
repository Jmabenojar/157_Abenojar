close all; clear all; clc;

lambda = 632.8e-9; %wavelength
cp = 5.2e-6;          %pixel pitch
k = 2*pi/lambda;      %wavenumber             

num = 10;              %number of images
% Hi sir palitan niyo lang po yung 
%'D:\Darkroom\Joshua\' to directory ng folder niyo
root='D:\Darkroom\Joshua\mainlatest\'; % <--- PAPALITAN TO
%% dz=3mm data
% root_from ='D:\Darkroom\Joshua\157 final project\expdata\u';     %file root to get images
root_from = 'D:\Darkroom\Joshua\157 final project\expdata\u';
firstplane=1;% First plane used
z0 = 44e-3;             % first plane distance of the original data
z0= 33e-3; %simulated data 1st distance
dz = 3e-3;             %distance between measurement planes (experimental)
z0=z0+dz*(firstplane-1); % ADJUSTED first plane distance depending on selected firstplane
% dz=dz*1.10;              % Adding error to the measurement plane distances

%% dz=12mm data
% firstplane=1         % First plane used
% root_from = [root 'FRCV\exp_data\exp_data12mm' num2str(firstplane) '\u'];     %file root to get images
% z0 = 44e-3; %z0 if first plane is 1
% dz=3e-3;       % Default dz
% z0=z0+dz*(firstplane-1); % z0 based on the first plane            
% dz = 12e-3;             %distance between measurement planes (experimental)
% dz=dz*1.05 %Adding error to the measurement plane distances

%% Folders for reconstructced images(Don't edit)
root_to1 = 'D:\Darkroom\Joshua\157 final project\sbmirresults'; % RS Convolution method
ftype = '.bmp';       %file type

%% Read intensities
centr = [550 670]; %locate cropping center (experimental data)
arrysize = 600; %desired cropped array size 

uu = zeros(2*floor(arrysize/2),2*floor(arrysize/2),num); %create an empty array to save intensities

for ii = 1:num % Read intensities
   int = (imread([root_from int2str(firstplane+ii-1) ftype])); 
   int = (double(int)); %convert to double precision
   % uu(:,:,ii)=int;
   uu(:,:,ii) = crp(int,centr,arrysize); %comment if cropping is unnecessary
   % figure(101); imagesc(uu(:,:,ii)); colormap(gray(255)); axis image;
end
amps = sqrt(uu); %calculate the amplitude from intensity(Input ng SBMIR)



%% Comparison
%smoothing function for display
smth = @(ph,n) atan2(conv2(sin(ph),ones(n)/2,'same'),conv2(cos(ph),ones(n)/2,'same'));

iter = 500; %set the number of iterations(CHOOSE DIVISIBLE BY 4)

load r2; % load a guess phase (better for experimental data)
guessphase = padarray(phase, [arrysize/2-512/2 arrysize/2-512/2], 'both'); 

%% SBMIR
[u_rec_sbmir,tFB,mse_amp_fb,mse_ph_fb] = sbmir(lambda,cp,dz,z0,num,iter,amps,guessphase,root_to1,'as'); % execute sbmir
disp(['ASM: Iterations: ',num2str(length(mse_amp_fb)-1), ...
    ' ; time: ',num2str(tFB), ' s']); %display execution time

 % get the reconstructed amplitude and phase
ampFB = abs(u_rec_sbmir); 
phFB = angle(u_rec_sbmir);
%crop
ampFB = ampFB; 
phFB = smth(phFB,10); 

%% Save plots to excel sheets
delete('mseplots_amp_sbmir.xlsx');
% Amplitude plots
xlswrite('mseplots_amp_sbmir.xlsx',transpose(1:length(mse_amp_fb)-1),'Sheet1',['A2:A' num2str(length(mse_amp_fb))]);
xlswrite('mseplots_amp_sbmir.xlsx',mse_amp_fb(2:length(mse_amp_fb)),'Sheet1','B2');
% Phase plots
delete('mseplots_ph_sbmir.xlsx');
xlswrite('mseplots_ph_sbmir.xlsx',transpose(1:length(mse_ph_fb)-1),'Sheet1',['A2:A' num2str(length(mse_ph_fb))]);
xlswrite('mseplots_ph_sbmir.xlsx',mse_ph_fb(2:length(mse_ph_fb)),'Sheet1','B2');
