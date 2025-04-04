def create_filtered_optimizer(optimizer_class, params, **optimizer_kwargs):
    # Filter parameters: only include parameters that aren't 2D
    filtered_params = []
    
    for p in params:
        if p.requires_grad and len(p.shape) != 2:
            filtered_params.append(p)
    
    # Create optimizer with filtered parameters
    if filtered_params == []:
        return None
    return optimizer_class(filtered_params, **optimizer_kwargs)


def create_2D_filtered_optimizer(optimizer_class, params, **optimizer_kwargs):
    # Filter parameters: only include parameters that are 2D
    filtered_params = []
    
    for p in params:
        if p.requires_grad and len(p.shape) == 2:
            filtered_params.append(p)
    
    # Create optimizer with filtered parameters
    if filtered_params == []:
        return None
    return optimizer_class(filtered_params, **optimizer_kwargs)