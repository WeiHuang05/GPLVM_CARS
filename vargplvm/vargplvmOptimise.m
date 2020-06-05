function [model, grChek] = vargplvmOptimise(fn, model, display, iters, varargin)


options = optOptions;
params = vargplvmExtractParam(model);

options(2) = 0.1*options(2);
options(3) = 0.1*options(3);

if display
    options(1) = 1;
 %   if length(params) <= 100
 %       options(9) = 1;
 %   end
end
options(14) = iters;

if iters > 0
    if isfield(model, 'optimiser') && ~isa(model.optimiser, 'function_handle')
        if isfield(model, 'optimiser')
            optim = str2func(model.optimiser);
        else
            optim = str2func('scg');
        end
        
        if strcmp(func2str(optim), 'optimiMinimize')
            % Carl Rasmussen's minimize function
            params = optim('vargplvmObjectiveGradient', params, options, model);
        elseif strcmp(func2str(optim), 'scg2')
            % NETLAB style optimization with a slight modification so that an
            % objectiveGradient can be used where applicable, in order to re-use
            % precomputed quantities.
            params = optim(fn, model, 'vargplvmObjectiveGradient', params,  options,  'vargplvmGradient', model);
        elseif strcmp(func2str(optim), 'sgd')
            params = optim(fn, model,  params,  options,  'vargplvmGradient', model);
        else
            % NETLAB style optimization.
            params = optim('vargplvmObjective', params,  options,  'vargplvmGradient', model);
        end
    elseif isfield(model, 'optimiser') && isa(model.optimiser, 'function_handle')
        f = fcnchk(model.optimiser);
        params = f(model);
    else
        error('vargplvmOptimise: Invalid optimiser setting.');
    end
    %model = vargplvmExpandParam(model, params);
    model = vargplvmExpandParam(model, params);
    
    % Check if SNR of the optimised model is reasonable (ortherwise a
    % bad local minimum might have been found)
    if isfield(model, 'throwSNRError')
        svargplvmCheckSNR({vargplvmShowSNR(model)}, [], [], model.throwSNRError);
    else
        svargplvmCheckSNR({vargplvmShowSNR(model)});
    end
end