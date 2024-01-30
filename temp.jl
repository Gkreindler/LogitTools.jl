t = Template(;
           user="Gkreindler",
           authors=["Gabriel Kreindler"],
           plugins=[
               License(name="MIT"),
               Git(),
               GitHubActions(),
           ],
       )

t("LogitTools.jl")
