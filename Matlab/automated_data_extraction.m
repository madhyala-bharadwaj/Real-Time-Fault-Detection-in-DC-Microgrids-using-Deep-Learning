if exist('data.mat', 'file') == 2
    load('data.mat', 'data');
    if ~iscell(data)
        data = {};
    end
else
    data = {};
end
save('data.mat', 'data');

R_line_values = 0.095:0.03:0.185;
L_line_values = 0.145:0.02:0.225;

iterationCount = 0;
for r_val = R_line_values
    for l_val = L_line_values
        iterationCount = iterationCount + 1;
        
        modelFile = 'Main_DC_MG.m';
        fileStr = fileread(modelFile);
        
        patternR = '(^\s*R_line\s*=\s*)[0-9\.Ee+-]+(;.*)';
        replacementR = sprintf('$1%.3f$2', r_val);
        fileStr = regexprep(fileStr, patternR, replacementR, 'lineanchors');
        
        patternL = '(^\s*L_line\s*=\s*)[0-9\.Ee+-]+(;.*)';
        replacementL = sprintf('$1%.3e$2', l_val*1e-3);
        fileStr = regexprep(fileStr, patternL, replacementL, 'lineanchors');
        
        fid = fopen(modelFile, 'w');
        if fid == -1
            error('Could not open %s for writing.', modelFile);
        end
        fwrite(fid, fileStr);
        fclose(fid);
        
        run(modelFile);
        
        simOut = sim('Model_DC_MG_Main.slx');

        load('V_B.mat', 'Vb');
        load('I_B.mat', 'Ib');
        load('labels.mat', 'label'); 
        
        commonRow = Vb(1, :);
        VbUnique = Vb(2, :);
        IbUnique = Ib(2, :);
        labelUnique = label(2, :);
        
        iterationData = {commonRow, VbUnique, IbUnique, labelUnique};
        
        load('data.mat', 'data');
        data = [data; iterationData];
        save('data.mat', 'data');
        
        fprintf('Completed simulation iteration %d: R_line = %.3f, L_line = %.3e\n', iterationCount, r_val, l_val*1e-3);
    end
end