function [x_train_set,y_train_set]=Read_Feat(train_root,pos_num,neg_num)
    x_train_set=zeros(pos_num+neg_num,13);
    y_train_set=zeros(pos_num+neg_num,1);
    y_train_set(1:pos_num,1)=1;
    for ii=1:pos_num
        read_tmp=load([train_root,'positive/',num2str(ii-1),'/feat.mat']);
        x_train_set(ii,:)=read_tmp.feat;
    end
    for ii=1:neg_num
        read_tmp=load([train_root,'negative/',num2str(ii-1),'/feat.mat']);
        x_train_set(ii+pos_num,:)=read_tmp.feat;
    end
end