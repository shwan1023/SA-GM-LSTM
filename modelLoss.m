function [loss,gradients] = modelLoss(net,X,T)

    % Forward data through the dlnetwork object.
    Y = forward(net,X);

    % Compute loss.
    loss = crossentropy(Y,T);

    % Compute gradients.
    gradients = dlgradient(loss,net.Learnables);

end
