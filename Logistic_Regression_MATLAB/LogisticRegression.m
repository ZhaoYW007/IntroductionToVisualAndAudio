function [w,b]=LogisticRegression(opt_type,x_train_set,y_train_set,...
    irratation_time,step_size)
    % opt=0,Gradient Descend;opt=1,Newton.
    if(opt_type==0)
        [w,b]=LogisticRegression_GD(x_train_set,y_train_set,...
            irratation_time,step_size);
    else
        [w,b]=LogisticRegression_Newton(x_train_set,y_train_set,...
            irratation_time,step_size);
    end
end

function [w,b]=LogisticRegression_GD(x_train_set,y_train_set,...
    irratation_time,step_size)
    w=10*rand(1,13)-5;
    b=10*rand-5;
    [ele_num,~]=size(y_train_set);
    for ii=1:irratation_time
        [gw,gb]=gradient(w,b,x_train_set,y_train_set,ele_num);
        w=w+gw*step_size;
        b=b+gb*step_size;
    end
end

function [w,b]=LogisticRegression_Newton(x_train_set,y_train_set,...
    irratation_time,step_size)
    w=10*rand(1,13)-5;
    b=10*rand-5;
    [ele_num,~]=size(y_train_set);
    
    
end

function [gw,gb]=gradient(w,b,x_train_set,y_train_set,ele_num)
    gw=zeros(1,13);
    gb=0;
    for ii=1:ele_num
        tmp_exp=exp(b+w*x_train_set(ii,:)');
        tmp_coef=y_train_set(ii,1)-tmp_exp/(1+tmp_exp);
        gw=gw+tmp_coef.*x_train_set(ii,:);
        gb=gb+tmp_coef;
    end
end
