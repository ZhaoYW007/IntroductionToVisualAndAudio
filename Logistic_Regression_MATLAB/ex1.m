train_root='./dataset/train/';
pos_num=100;
neg_num=100;

[x_train_set,y_train_set]=Read_Feat(train_root,pos_num,neg_num);
[w0,b0]=LogisticRegression(0,x_train_set,y_train_set,100000,0.001);
test0=zeros(pos_num+neg_num,1);
for ii=1:pos_num+neg_num
    y_=1/(1+exp(-(w0*x_train_set(ii,:)'+b0)));
    test0(ii,1)=(y_>=0.5);
end
score0=sum(test0==y_train_set)/200;
%[w1,b1]=LogisticRegression(1,x_train_set,y_train_set);