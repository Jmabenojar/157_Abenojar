dircontent = dir('*.bmp');
for fileidx = 1: numel(dircontent);
   filename = dircontent(fileidx).name;
   [img, map, trans] = imread(filename);
   if ~isempty(trans)
       warning('transparency information of %s discarded', filename);
   end
   [~, basename] = fileparts(filename);
   outname = [basename, '.png'];
   if isempty(map)
      %greyscale and colour image
      imwrite(img, outname, 'png');
   else
      imwrite(img, map, outname, 'png');
   end
end