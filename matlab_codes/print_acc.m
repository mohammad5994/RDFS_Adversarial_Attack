for jj=1:49
sprintf("%d",jj)
acc0=load('/home/ehsan/fliipimg_mat_files/5_percent/5/results_10.mat','accuracy');
acc0=acc0.accuracy(1);

%acc=sprintf('/home/ehsan/fliipimg_mat_files/5_percent/5/results_%d.mat',jj);
%acc=load(acc,'accuracy');
%accc(jj+1)=acc.accuracy(1);

end