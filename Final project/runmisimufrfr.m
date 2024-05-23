%% SINGLE BEAM MULTIPLE-INTENSITY RECONSTRUCTION (Conventional vs. unordered algorithm)
%  Using simulated data
%  Binamira, Jonel F.
%  February 27, 2019 

% (Note: all units are in meters)

clear all; close all; clc;

%%  Adding the functions needed
% Locates the 'sbmir_functions' and 'sim_images' folder to add the sbmit
% functions in those folders

% addpath(genpath('D:\Darkroom\Jonel\SBMIR files\sim_functions'));
% addpath(genpath('sbmir_functions'));
% addpath(genpath('sim_images'));

%% Generating the object field	

% Loading the image as amplitude
% img =  normed(double((imread('square8.bmp'))),2);
img =  normed(double((imread('square8.bmp'))),2);

% Initializing the parameters
ij = sqrt(-1);  % Imaginary number
[M,N] = size(img); % Array size of the object
dx = 5.2e-6; % x pixel size (camera pitch)
dy = dx; % y pixel size (camera pitch)
cp=dx;
lambda = 632.8e-9; % Wavelength
L = M*dx; % Array side length
z_threshold=dx*L/lambda;
disp(z_threshold)
Planes = 5; % Number of observation planes
iter=400;
dz = 3e-3; % Distance between measurement planes

% Generating a diffuser (random array)
% rand_arry = rand(M,N);
load('rawr.mat');
% Setting the coordinates in spatial domain
[X,Y] = meshgrid(-M/2:M/2-1,-M/2:M/2-1);
X = normed(X,1); Y = normed(Y,1);
	
% Initializing the object phase (spherical phase)
R = 352; % height of spherical wave (related to the focal length)
path = (R.^2-(Y).^2-(X).^2);
path_x2 = ((R.^2-(Y).^2-(X).^2))./max(max(R.^2-(Y).^2-(X).^2));
U = (2*pi/lambda)*path_x2;
	
% Simulating the object field (with diffuse ilumination)
randsx = pi; % Diffuser depth of randomization
U = U + rand_arry*randsx; % Add noise to phase
uu = img.*exp(ij*U); % Object field

figure(1); colormap(gray(256));
subplot 121; imagesc(abs(uu)); axis image; title('Object amplitude')
subplot 122; imagesc(angle(uu)); axis image; title('Object phase')
smth = @(ph,n) atan2(conv2(sin(ph),ones(n)/2,'same'),conv2(cos(ph),ones(n)/2,'same'));
imwrite(uint8(normalize(smth(angle(uu),10))),'phase.png')

%% Generating intensity measurements

u = zeros(M,M,Planes);      % Store cropped intensity images here
z0 = 35e-3;    % Distance between plane 1 and the object
dist = zeros(1,Planes); % Store intensity image distances here
ps=0;
% ps = 400; % Pad size
u = padarray(u,[ps,ps],0,'both'); % Places pad on intensity image array

for ii = 1:Planes
    dist(ii) = z0+(ii-1)*dz; % Propagation distance of the (ii)th plane
    % u(ps+1:M+ps,ps+1:M+ps,ii) = abs(prop_TF(uu,M*delx,lambda,dist(ii))).^2; % Propagation of object to (ii)th plane
    u(ps+1:M+ps,ps+1:M+ps,ii) = abs(proppp(uu,lambda,dist(ii),dx,'ir')).^2;
	u(:,:,ii) = double(uint8(normed(u(:,:,ii),2).*255));
    figure(2); colormap(gray(256)); imagesc(abs(u(ps+1:M+ps,ps+1:M+ps,ii))); axis off; axis image; title(['Intensity image ' num2str(ii)])
    imwrite(uint8(u(ps+1:M+ps,ps+1:M+ps,ii)),['D:\Darkroom\Joshua\157 final project\results\r' num2str(ii),'.bmp'])
end; clear ii;
u = u(ps+1:M+ps,ps+1:M+ps,:); % Removes the pad on intensity image array
amps = sqrt(u); % Amplitude


k = 2*pi/lambda;      %wavenumber             

num = Planes;              %number of images
% Hi sir palitan niyo lang po yung 
%'D:\Darkroom\Joshua\' to directory ng folder niyo
root='D:\Darkroom\Joshua\mainlatest\'; % <--- PAPALITAN TO
%% dz=3mm data
% root_from = [root 'SIMULATEDuniverse\simulateddata\s'];     %file root to get images

%% dz=12mm data
% firstplane=1         % First plane used
% root_from = [root 'FRCV\exp_data\exp_data12mm' num2str(firstplane) '\u'];     %file root to get images
% z0 = 44e-3; %z0 if first plane is 1
% dz=3e-3;       % Default dz
% z0=z0+dz*(firstplane-1); % z0 based on the first plane            
% dz = 12e-3;             %distance between measurement planes (experimental)
% dz=dz*1.05 %Adding error to the measurement plane distances

%% Folders for reconstructced images(Don't edit)
root_to1 = [root 'FRCV-results\sim\AS']; % RS Convolution method
% root_to2 = [root 'FRCV-results\TF'];
root_to3 = [root 'FRCV-results\sim\IR']; % Proposed method
ftype = '.bmp';       %file type

%% Read intensities
% centr = [550 670]; %locate cropping center (experimental data)
% arrysize = 800; %desired cropped array size 

uu = zeros(640,640,num); %create an empty array to save intensities

arrysize=640;


%% Comparison
%smoothing function for display
smth = @(ph,n) atan2(conv2(sin(ph),ones(n)/2,'same'),conv2(cos(ph),ones(n)/2,'same'));

% iter = 800; %set the number of iterations(CHOOSE DIVISIBLE BY 4)

load r2; % load a guess phase (better for experimental data)
guessphase = padarray(phase, [640/2-512/2 640/2-512/2], 'both'); 
% figure(101); imagesc(guessphase); colormap(gray); axis image;
%% SBMIR
[u_rec_sbmir,tFB,mse_amp_fb,mse_ph_fb] = sbmir(lambda,cp,dz,z0,num,iter,amps,guessphase,root_to1,'as'); % execute sbmir
disp(['ASM: Iterations: ',num2str(length(mse_amp_fb)-1), ...
    ' ; time: ',num2str(tFB), ' s']); %display execution time

 % get the reconstructed amplitude and phase
ampFB = abs(u_rec_sbmir); 
phFB = angle(u_rec_sbmir);
%crop
ampFB = crp(ampFB,[arrysize/2 arrysize/2],600); 
phFB = crp(smth(phFB,10),[arrysize/2 arrysize/2],600); 

%% TF
% [u_rec_sbmir2,tFB2,mse_amp_fb2,mse_ph_fb2] = sbmir(lambda,cp,dz,z0,num,iter,amps,phase,root_to2,'tf'); % execute sbmir
% disp(['TF: Iterations: ',num2str(length(mse_amp_fb2)-1), ...
%     ' ; time: ',num2str(tFB2), ' s']); %display execution time
% 
% 
%  % get the reconstructed amplitude and phase
% ampFB2 = abs(u_rec_sbmir2); 
% phFB2 = angle(u_rec_sbmir2);
% %crop
% ampFB2 = crp(ampFB2,[arrysize/2 arrysize/2],600); 
% phFB2 = crp(smth(phFB2,10),[arrysize/2 arrysize/2],600); 
%% IR
[u_rec_sbmir3,tFB3,mse_amp_fb3,mse_ph_fb3] = sbmir_fc(lambda,cp,dz,z0,num,iter/2,amps,guessphase,root_to3,'ir'); % execute sbmir
disp(['IR: Iterations: ',num2str(length(mse_amp_fb3)-1), ...
    ' ; time: ',num2str(tFB3), ' s']); %display execution time

 % get the reconstructed amplitude and phase
ampFB3 = abs(u_rec_sbmir3); 
phFB3 = angle(u_rec_sbmir3);
%crop
ampFB3 = crp(ampFB3,[arrysize/2 arrysize/2],600); 
phFB3 = crp(smth(phFB3,10),[arrysize/2 arrysize/2],600); 
%% Save plots to excel sheets
delete('mseplots_amp_sbmir.xlsx');
xlswrite('mseplots_amp_sbmir.xlsx',transpose(1:length(mse_amp_fb)-1),'Sheet1',['A2:A' num2str(length(mse_amp_fb))]);
xlswrite('mseplots_amp_sbmir.xlsx',mse_amp_fb(2:length(mse_amp_fb)),'Sheet1','B2');
xlswrite('mseplots_amp_sbmir.xlsx',mse_amp_fb3(2:length(mse_amp_fb3)),'Sheet1','C2');
% xlswrite('mseplots_amp_sbmir.xlsx',mse_amp_fb2(2:length(mse_amp_fb2)),'Sheet1','D2');
% Phase plots
delete('mseplots_ph_sbmir.xlsx');
xlswrite('mseplots_ph_sbmir.xlsx',transpose(1:length(mse_ph_fb)-1),'Sheet1',['A2:A' num2str(length(mse_ph_fb))]);
xlswrite('mseplots_ph_sbmir.xlsx',mse_ph_fb(2:length(mse_ph_fb)),'Sheet1','B2');
xlswrite('mseplots_ph_sbmir.xlsx',mse_ph_fb3(2:length(mse_ph_fb3)),'Sheet1','C2');
% xlswrite('mseplots_ph_sbmir.xlsx',mse_ph_fb2(2:length(mse_ph_fb2)),'Sheet1','D2');

%% Display results
figure(4),
% subplot(211),imshow(mat2gray(phFB)),axis image; colormap(gray(255)); title('AS')
% subplot(212),imshow(mat2gray(phFB3)),axis image; colormap(gray(255)); title('IR')
% figure(1),
subplot 221, imshow(mat2gray(ampFB)); axis image; colormap(gray(255)); title('AMP: SBMIR');
subplot 222, imshow(mat2gray(phFB));  axis image; title('PH: SBMIR');
subplot 223, imshow(mat2gray(ampFB3));  axis image; colormap(gray(255)); title('AMP: SBMIR-F');
subplot 224, imshow(mat2gray(phFB3));  axis image; title('PH: SBMIR-F');
% 
figure(5),hold on
plot(2:length(mse_amp_fb),mse_amp_fb(2:length(mse_amp_fb)), ...
    'b','Marker','o','Linewidth',2,'LineStyle','-');
plot(2*(2:length(mse_amp_fb3)),mse_amp_fb3(2:length(mse_amp_fb3)), ...
    'r','Marker','*','LineStyle','--')
set(gca,'FontSize',28);
xlabel('Iteration','FontSize',28), ylabel('Amplitude MSE','FontSize',28), xlim([0,iter]); 
% ylim([min(mse_amp_ASC(2:length(mse_amp_ASC))),max(mse_amp_fb)])
legend('SBMIR','SBMIR-F(compensated)','Location', 'NorthEast','Orientation','vertical'); 
% % saveas(gcf, [root_to,'\AMSEplot_iter=',num2str(iter),'_num=',num2str(num),'.png']); %save figure

figure(6),hold on
plot(2:length(mse_ph_fb),mse_ph_fb(2:length(mse_ph_fb)), ...
    'b','Marker','o','Linewidth',2,'LineStyle','-');
plot(2*(2:length(mse_ph_fb3)),mse_ph_fb3(2:length(mse_ph_fb3)), ...
    'r','Marker','*','LineStyle','--')
set(gca,'FontSize',28);
xlabel('Iteration','FontSize',28), ylabel('Phase MSE','FontSize',28), xlim([0,iter]); 
% ylim([min(mse_ph_ASC(2:length(mse_ph_ASC))),max(mse_ph_ASC)])
legend('SBMIR','SBMIR-F(compensated)','Location', 'NorthEast','Orientation','vertical'); 
% clear all; close all;

%% display iterations 10, 50, and 170
resultroot=[root 'FRCV-results\sim\'];

imgarray=zeros([300,300,6]); %Empty array

AS1=['AS\Ph_num=' num2str(num) '_iter=50.bmp'];
AS2=['AS\Ph_num=' num2str(num) '_iter=' num2str(iter/2) '.bmp'];
AS3=['AS\Ph_num=' num2str(num) '_iter=' num2str(iter) '.bmp'];
% TF1='TF\PhFB_num=5_iter=10.bmp';
% TF2='TF\PhFB_num=5_iter=50.bmp';
% TF3='TF\PhFB_num=5_iter=170.bmp';
IR1=['IR\PhFB_num=' num2str(num) '_iter=50.bmp'];
IR2=['IR\PhFB_num=' num2str(num) '_iter=' num2str(iter/2) '.bmp'];
IR3=['IR\PhFB_num=' num2str(num) '_iter=' num2str(iter) '.bmp'];
imgarray(:,:,1)=double(imread([resultroot AS1]));
imgarray(:,:,2)=double(imread([resultroot AS2]));
imgarray(:,:,3)=double(imread([resultroot AS3]));
% imgarray(:,:,4)=double(imread([resultroot TF1]));
% imgarray(:,:,5)=double(imread([resultroot TF2]));
% imgarray(:,:,6)=double(imread([resultroot TF3]));
imgarray(:,:,4)=double(imread([resultroot IR1]));
imgarray(:,:,5)=double(imread([resultroot IR2]));
imgarray(:,:,6)=double(imread([resultroot IR3]));

rawr=[iter,50,iter/2];
figure(7)
for i=1:6
    subplot(2,3,i), imagesc(imgarray(:,:,i)), axis image; axis off; colormap(gray(255));
    if i<4
        title(['AS iter' num2str(rawr(mod(i,3)+1))])
    end
    if i>3
        title(['IR iter' num2str(rawr(mod(i,3)+1))])
    end
end