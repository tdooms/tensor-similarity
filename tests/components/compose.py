# %%
%load_ext autoreload
%autoreload 2

from components import Linear, MLP, Head, sequential, parallel
# %%
mlp = MLP(10, 10)
embed = Linear(7, 10)
head = Head(10, 5)
# %%
parallel(mlp.network(), n=2).draw()
# %%
parallel(embed.network(), n=3).draw()
# %%
sequential(embed.network(), mlp.network()).draw()
# %%
sequential(embed.network(), mlp.network(), mlp.network(), mlp.network(), head.network()).draw(show_tags=False, color=['N', 'E', 'B', 'U'], legend=False)