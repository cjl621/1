%fig = uifigure;
%s = uislider(fig);
fig = uifigure;
sld = uislider(fig);
sld.Limits = [0 0.00001];
sld.Value = 0.000005;

a = sld.Value;
x1 = 5000;
x2 = 7000;
k = 1000;
t = 0:1:k;
d = 0.005;%精度参数

x3 = 0:1:k;
x4 = 0:1:k;

k1 = 2;

x3(1) = x1;
x4(1) = x2; 

while(k1<k)

x3(k1) = x3(k1-1) + (x3(k1-1)*(0.05*x3(k1-1)*(1 - x3(k1-1)/150000) - a*x3(k1-1)*x4(k1-1))/100)*d;

x4(k1) = x4(k1-1) + (x4(k1-1)*(0.08*x4(k1-1)*(1 - x4(k1-1)/400000) - a*x3(k1-1)*x4(k1-1))/100)*d;

k1 = k1+1;

end

fig = uifigure;
sld = uislider(fig);
sld.Limits = [0 0.00001];
sld.Value = 0.000005;

title "x3(t)"
plot(t,x4);   
    %%图片不能传上来，效果成功，调整精度值可以获得更严格的结果

