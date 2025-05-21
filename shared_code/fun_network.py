#%%
# =============================================================================
# Network analysis functions    
# =============================================================================

def sort_modularity(fc):
    """Eliminate it soon, peplace by 'allegiance_matrix_analysis' in metaconnectivity"""
    #Modularity of Louvain
    # modules, louvain = bct.modularity.modularity_louvain_dir(fc)
    # modules, louvain = bct.modularity.modularity_louvain_dir(fc)
    modules, louvain = bct.modularity.modularity_louvain_und_sign(fc, gamma=1.1)
    # print(np.unique(modules),louvain)
    
    #sort accord the modularity
    sort_modules = np.argsort(modules)
    # print(sort_modules)
    fc_mod = fc[:,sort_modules][sort_modules,:] #fc sorted by modularity
    
    return fc_mod

# =============================================================================

