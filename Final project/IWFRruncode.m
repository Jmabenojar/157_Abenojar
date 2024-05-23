%% Initialize parameters
lambda = 632e-9;
cp = 5.2e-6;
dist = 3e-3;
z0 = 44e-3;
% z0=35e-3
num = 10;
iter = 500;
arrysize=600;
dist_array = z0:dist:z0+(num-1)*dist;
root_to='D:\Darkroom\Joshua\157 final project\results\';
%% read intensity recordings
uu = zeros(2*floor(arrysize/2),2*floor(arrysize/2),num);
for ii = 1:num % Read intensities
   int = (imread(['D:\Darkroom\Joshua\157 final project\expdata\u' int2str(ii) '.bmp'])); 
   int = (normalize(double(int))); %convert to double precision
   % uu(:,:,ii)=int;
   uu(:,:,ii) = crp(int,[550 670],600); %comment if cropping is unnecessary
   figure(101); imagesc(uu(:,:,ii)); colormap(gray(255)); axis image;
end
amps = sqrt(uu); %calculate the amplitude from intensity(Input ng SBMIR)

%% Create guess phase
load r2
gph = padarray(phase, [600/2-512/2 600/2-512/2], 'both');
[C,R] = size(amps(:,:,1)); %get aperture size
L = cp*C;  %side length
k = 2*pi/lambda;

ALLPLANEARRAY=zeros([C,R,num]);
OBJPLANEARRAY=zeros([C,R,num]);
%% set guess phase for all planes
ALLPLANEARRAY(:,:,1)=amps(:,:,1).*exp(j*gph);
for i=2:num
    ALLPLANEARRAY(:,:,i)=prop(ALLPLANEARRAY(:,:,1),lambda,dist*(i-1),cp,'ir');
    ALLPLANEARRAY(:,:,i)=amps(:,:,i).*exp(j*angle(ALLPLANEARRAY(:,:,i)));
end
smth = @(ph,n) atan2(conv2(sin(ph),ones(n)/2,'same'),conv2(cos(ph),ones(n)/2,'same'));
ampmse=[];
tic
%% Iterative part
for ii=1:iter
    %% Backpropagation from all  measurement planes
    for i=1:num
        OBJPLANEARRAY(:,:,i)=prop(ALLPLANEARRAY(:,:,i),lambda,-dist_array(i),cp,'ir');
    end
    %% Average of propagation from all measurement planes
    U_obj=mean(OBJPLANEARRAY,3);
    obj_phase=angle(U_obj);
    obj_amp=abs(U_obj);
    figure(2)
    subplot 131, imshow(mat2gray(obj_amp)); axis image; colormap(gray(255)); title(['AMPLITUDE' num2str(ii)]);
    subplot 132, imshow(mat2gray(smth(obj_phase,10)));  axis image; title(['PHASE' num2str(ii)]);
    %% Filtering(comment for no filter)
    % filter=lowpassf(U_obj,100);
    % filter=imbinarize(abs(filter).^2,0);
    % subplot 133, imshow(mat2gray(filter));  axis image; title(['Filter' num2str(ii)]);
    % if rem(ii,20)==0 || ii==5
    %     imwrite(uint8(normalize(filter)),[root_to,'Filter_iter=',num2str(ii),'.bmp'])
    % end
    % U_obj=U_obj.*filter;
    %% Propagate back to measurement planes
    for i=1:num
        ALLPLANEARRAY(:,:,i)=prop(U_obj,lambda,dist_array(i),cp,'ir');
        if i==1
            eramp=(abs(ALLPLANEARRAY(:,:,1))-amps(:,:,1)).^2;
            err_amp = mean(mean(eramp));
            ampmse=[ampmse;err_amp];
        end
        ph=angle(ALLPLANEARRAY(:,:,i));
        ALLPLANEARRAY(:,:,i)=amps(:,:,i).*exp(j*ph);
    end
    if rem(ii,1)==0 || ii==5
        imwrite(uint8(normalize(obj_amp)),[root_to,'\Amp_iter=',num2str(ii),'.bmp']);
        imwrite(uint8(normalize(smth(obj_phase,10))),[root_to,'\Ph_iter=',num2str(ii),'.bmp']);
    end
end
xlswrite('mse.xlsx',transpose(1:length(ampmse)),'Sheet1',['A2:A' num2str(iter+1)]);
xlswrite('mse.xlsx',sqrt(ampmse(1:length(ampmse))),'Sheet1',['B2:B' num2str(iter+1)]);
tFB = toc;