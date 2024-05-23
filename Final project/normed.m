function xx = normed(x,opt);
switch opt
    case 1
        xx = x./max(max(x));
    case 2
        xx = (x-min(min(x)))./max(max(x));
    case 3
        xmi = x-min(min(x));
        xx = xmi./(max(max(xmi)));
end