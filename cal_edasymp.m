function pband=cal_edasymp(data_down,fs_d)
%% spec eda 
% filter 
cutoff_low=0.5;
cutoff_high=0.01;
order=2;
[B_low,A_low] = butter(order,2*cutoff_low/fs_d,'low');
low_passed = filtfilt(B_low,A_low,data_down);

[B_high,A_high] = butter(order,2*cutoff_high/fs_d,'high');
filt_data = filtfilt(B_high,A_high,data_down);
% apply spec 
window=blackman(fs_d*60);
overlap=fs_d*59;
[s,w,time]=spectrogram(filt_data,window,overlap,[],fs_d);


a=find(w<=0.045);
b=find(w>=0.25);
f1=a(end);
f2=b(1);
freq=w([f1:f2]);
% mm=zeros(size(s,2),1);
pband=zeros(size(s,2),1);

for i=1:size(s,2)
sig=s([f1:f2],i);

% mm(i,1)=mean(abs(sig).^2);
pband(i,1) = bandpower(abs(s(:,i)),w,[0.045 0.25],'psd');

end
kk=nan(1,60); %% I am not sure check with Albe
pband=[kk';pband];
end
