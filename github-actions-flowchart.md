# GitHub actions diagram

This is a collection of flowcharts for all the GitHub actions used across the COAsT and COAsT-site repos
## COAsT
### building Packages
```mermaid
graph LR;    
    subgraph publish_package - runs on push to master
    A1[Setup python]-- 3.8 -->B1;    
    B1[Install dependencies]-->C1;
    C1[Setup Enviroment]-->D1;
    D1[Build package]-->E1;
    E1[Test Package Install]-->F1
    F1[Publish to pypi]-->G1
    G1[Generate Conda Metadata]-->H1
    H1[Publish to Anaconda]
    end;
    
    subgraph build_package - runs on push to non-master 
    A[Setup python]-- 3.8 and 3.9 -->B;    
    B[Install dependencies]-->C;
    C[Setup Enviroment]-->D;
    D[Build package]-->E;
    E[Test Package Install]-->F
    F[Generate Conda Metadata] 
    end;
```

### Verification and Formatting
```mermaid
graph LR

    subgraph formatting - runs on pull requests
    A[Setup python]-- 3.9 -->B;    
    B[Install black]-->C;
    C[Check formatting]--> D;
    D[Apply formatting]
    end;
    
    subgraph verifiy_package - runs for every push
    A1[Setup python]-- 3.8 and 3.9 -->B1;    
    B1[Install dependencies]-->C1;
    C1[Lint]-->D1;
    D1[Test]
    end;
    click B1 "https://www.github.com" "tooltip"
```

### interactions with other repos
```mermaid
flowchart LR
    subgraph b1[push_notebooks - runs on push to develop]
        direction LR
        subgraph b2[COAsT site - markdown ]
            direction TB
            a[checkout docsy site] -->b
            b[checkout coast] -->c
            c[add python] -->d
            d[covert notebooks] -->e
            e[commit changes]            
         end
    t[Repository Dispatch] -- event pushed --> b2    
    end
    click a "https://github.com/British-Oceanographic-Data-Centre/COAsT-site" "Docsy site for COAsT repo"
```
```mermaid
flowchart LR
    subgraph b3[push_docstrings - runs on push to develop]
        direction LR
        subgraph b4[COAsT site - docstrings ]
            direction TB
            a1[checkout docsy site] -->b1
            b1[checkout coast] -->c1
            c1[add python] -->d1
            d1[covert docstrings] -->e1
            e1[commit changes]            
        end
    r[Repository Dispatch] -- event pushed --> b4       
    end
    click a1 "https://github.com/British-Oceanographic-Data-Centre/COAsT-site" "Docsy site for COAsT repo"
```

## COAsT-site
These are the actions used on the COAsT-site repo. 

### Convert to markdown
See [Interactions with other repos](#interactions-with-other-repos) for the related markdown and docstring workflows

### Build site
```mermaid
graph LR
    subgraph hugo - runs on push to master
    A[checkout site]-->B;    
    B[Setup Hugo] -- v0.70.0 -->C;
    C[Setup Nodejs]-- v12 --> D;
    D[Build]-->E
    E[Deploy]
    end;
```