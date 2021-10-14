%fig = uifigure;
%s = uislider(fig);
%fig = uifigure;
%sld = uislider(fig);
%sld.Limits = [0 0.00001];
%sld.Value = 0.000005;

a = 0.0000001;
x1 = 5000;
x2 = 7000;
k =4000;
t = 0:1:k;
d = 0.005;%精度参数

x3 = 0:1:k;
x4 = 0:1:k;

k1 = 2;

x3(1) = x1;
x4(1) = x2; 

while(k1<k+2)

x3(k1) = x3(k1-1) + (x3(k1-1)*(0.05*x3(k1-1)*(1 - x3(k1-1)/150000) - a*x3(k1-1)*x4(k1-1))/100)*d;

x4(k1) = x4(k1-1) + (x4(k1-1)*(0.08*x4(k1-1)*(1 - x4(k1-1)/400000) - a*x3(k1-1)*x4(k1-1))/100)*d;

k1 = k1+1;

end


%plot(t,x3,'LineWidth',2);
%hold on;
%plot(t,x4,'LineWidth',2);  
    %%图片不能传上来，效果成功，调整精度值可以获得更严格的结果
    %%（x3,x4）图像稳定在（35000，400000）点
    %%可以绘制出x3,x4的图像，数值正确，
    
[x,y]=meshgrid(linspace(-600000,600000));
streamslice(x,y,x.*(-0.05/150000*x+0.05-a*y),y.*(-0.08/400000*y+0.08+a*x));
xlabel('x');ylabel('y');    
    
    
    
    
    
    
    
    
    