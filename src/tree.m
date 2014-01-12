function [ files ] = tree( path )

all_files = sub_tree(path);
files = cell(size(all_files));
for i = 1 : length(all_files)
    files{i} = all_files(i).name;
end

end

function [ all_files ] = sub_tree( path )

all_files = dir(path);
all_files = all_files(3:end);

i = 1;

while i <= length(all_files)
    all_files(i).name = fullfile(path,all_files(i).name);
    if all_files(i).isdir
        sub_files = sub_tree(all_files(i).name);
        all_files = [all_files(1:i-1); sub_files; all_files(i+1:end)];
        i = i + length(sub_files);
    else
        i = i + 1;
    end
end

end