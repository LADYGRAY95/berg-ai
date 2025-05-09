import pandas as pd
import os
from sklearn.model_selection import train_test_split

def create_gene_datasets():
    """
    Create training and testing datasets from the gene data in paste.txt
    """
    # Raw gene data from paste.txt
    gene_data = """gene_id,gene_symbol,pathway,disease_type,function,protein_level
ENSG00000133703,KRAS,MAPK,cancer,GTPase involved in signal transduction,high
ENSG00000132155,RAF1,MAPK,cancer,Serine/threonine-protein kinase,medium
ENSG00000169032,MAP2K1,MAPK,cancer,Dual specificity mitogen-activated protein kinase,high
ENSG00000100030,MAPK1,MAPK,cancer,Mitogen-activated protein kinase 1,high
ENSG00000102882,MAPK3,MAPK,cancer,Mitogen-activated protein kinase 3,medium
ENSG00000121879,PIK3CA,PI3K-AKT,cancer,"Phosphatidylinositol 4,5-bisphosphate 3-kinase catalytic subunit",high
ENSG00000142208,AKT1,PI3K-AKT,cancer,RAC-alpha serine/threonine-protein kinase,high
ENSG00000198793,MTOR,PI3K-AKT,cancer,Serine/threonine-protein kinase mTOR,medium
ENSG00000171862,PTEN,PI3K-AKT,cancer,"Phosphatidylinositol 3,4,5-trisphosphate 3-phosphatase",high
ENSG00000141510,TP53,p53,cancer,Cellular tumor antigen p53,high
ENSG00000135679,MDM2,p53,cancer,E3 ubiquitin-protein ligase Mdm2,medium
ENSG00000124762,CDKN1A,p53,cancer,Cyclin-dependent kinase inhibitor 1,high
ENSG00000087088,BAX,p53,cancer,Apoptosis regulator BAX,medium
ENSG00000125084,WNT1,WNT,cancer,Proto-oncogene Wnt-1,high
ENSG00000042980,FZD1,WNT,cancer,Frizzled-1,medium
ENSG00000134982,APC,WNT,cancer,Adenomatous polyposis coli protein,high
ENSG00000168036,CTNNB1,WNT,cancer,Catenin beta-1,high
ENSG00000130203,APOE,Protein_Misfolding,neurodegenerative,Apolipoprotein E,high
ENSG00000142192,APP,Protein_Misfolding,neurodegenerative,Amyloid-beta precursor protein,high
ENSG00000080815,PSEN1,Protein_Misfolding,neurodegenerative,Presenilin-1,medium
ENSG00000186868,MAPT,Protein_Misfolding,neurodegenerative,Microtubule-associated protein tau,high
ENSG00000232810,TNF,Neuroinflammation,neurodegenerative,Tumor necrosis factor,high
ENSG00000125538,IL1B,Neuroinflammation,neurodegenerative,Interleukin-1 beta,high
ENSG00000095970,TREM2,Neuroinflammation,neurodegenerative,Triggering receptor expressed on myeloid cells 2,medium
ENSG00000131095,GFAP,Neuroinflammation,neurodegenerative,Glial fibrillary acidic protein,high
ENSG00000114209,BECN1,Autophagy,neurodegenerative,Beclin-1,medium
ENSG00000057663,ATG5,Autophagy,neurodegenerative,Autophagy protein 5,medium
ENSG00000009954,TFEB,Autophagy,neurodegenerative,Transcription factor EB,medium
ENSG00000005156,LAMP2,Autophagy,neurodegenerative,Lysosome-associated membrane glycoprotein 2,high
ENSG00000185345,PARK2,Mitochondrial,neurodegenerative,E3 ubiquitin-protein ligase parkin,high
ENSG00000163993,PINK1,Mitochondrial,neurodegenerative,Serine/threonine-protein kinase PINK1,high
ENSG00000101787,DJ1,Mitochondrial,neurodegenerative,Protein/nucleic acid deglycase DJ-1,medium
ENSG00000142168,SOD1,Mitochondrial,neurodegenerative,"Superoxide dismutase [Cu-Zn]",high
ENSG00000159640,ACE,RAAS,cardiovascular,Angiotensin-converting enzyme,high
ENSG00000135744,AGT,RAAS,cardiovascular,Angiotensinogen,high
ENSG00000144891,AGTR1,RAAS,cardiovascular,Type-1 angiotensin II receptor,medium
ENSG00000180772,AGTR2,RAAS,cardiovascular,Type-2 angiotensin II receptor,low
ENSG00000084674,APOB,Lipid_Metabolism,cardiovascular,Apolipoprotein B-100,high
ENSG00000130164,LDLR,Lipid_Metabolism,cardiovascular,Low-density lipoprotein receptor,high
ENSG00000169174,PCSK9,Lipid_Metabolism,cardiovascular,Proprotein convertase subtilisin/kexin type 9,medium
ENSG00000113161,HMGCR,Lipid_Metabolism,cardiovascular,"3-hydroxy-3-methylglutaryl-coenzyme A reductase",high
ENSG00000136244,IL6,Inflammation,cardiovascular,Interleukin-6,high
ENSG00000232810,TNF,Inflammation,cardiovascular,Tumor necrosis factor,high
ENSG00000132693,CRP,Inflammation,cardiovascular,C-reactive protein,high
ENSG00000109320,NFKB1,Inflammation,cardiovascular,Nuclear factor NF-kappa-B p105 subunit,medium
ENSG00000164867,NOS3,Endothelial,cardiovascular,Nitric oxide synthase endothelial,high
ENSG00000078401,EDN1,Endothelial,cardiovascular,Endothelin-1,medium
ENSG00000112715,VEGF,Endothelial,cardiovascular,Vascular endothelial growth factor A,high
ENSG00000128052,KDR,Endothelial,cardiovascular,Vascular endothelial growth factor receptor 2,medium
ENSG00000204287,HLA-DRB1,MHC,autoimmune,HLA class II histocompatibility antigen,high
ENSG00000179344,HLA-DQB1,MHC,autoimmune,HLA class II histocompatibility antigen,high
ENSG00000234745,HLA-B,MHC,autoimmune,HLA class I histocompatibility antigen,high
ENSG00000206503,HLA-A,MHC,autoimmune,HLA class I histocompatibility antigen,high
ENSG00000112116,IL17A,Cytokine,autoimmune,Interleukin-17A,high
ENSG00000110944,IL23A,Cytokine,autoimmune,Interleukin-23 subunit alpha,medium
ENSG00000111537,IFNG,Cytokine,autoimmune,Interferon gamma,high
ENSG00000232810,TNF,Cytokine,autoimmune,Tumor necrosis factor,high
ENSG00000010610,CD4,T_Cell,autoimmune,T-cell surface glycoprotein CD4,high
ENSG00000153563,CD8A,T_Cell,autoimmune,T-cell surface glycoprotein CD8 alpha chain,high
ENSG00000163599,CTLA4,T_Cell,autoimmune,Cytotoxic T-lymphocyte protein 4,high
ENSG00000101017,CD28,T_Cell,autoimmune,T-cell-specific surface glycoprotein CD28,medium
ENSG00000177455,CD19,B_Cell,autoimmune,B-lymphocyte antigen CD19,high
ENSG00000156738,MS4A1,B_Cell,autoimmune,B-lymphocyte antigen CD20,high
ENSG00000105369,CD79A,B_Cell,autoimmune,B-cell antigen receptor complex-associated protein alpha chain,medium
ENSG00000211895,IGHM,B_Cell,autoimmune,Immunoglobulin heavy constant mu,high
ENSG00000129965,INS,Insulin,metabolic,Insulin,high
ENSG00000171105,INSR,Insulin,metabolic,Insulin receptor,high
ENSG00000169047,IRS1,Insulin,metabolic,Insulin receptor substrate 1,medium
ENSG00000105329,IRS2,Insulin,metabolic,Insulin receptor substrate 2,medium
ENSG00000106633,GCK,Glucose,metabolic,Glucokinase,high
ENSG00000131482,G6PC,Glucose,metabolic,Glucose-6-phosphatase catalytic subunit,high
ENSG00000124253,PCK1,Glucose,metabolic,Phosphoenolpyruvate carboxykinase,medium
ENSG00000159399,HK2,Glucose,metabolic,Hexokinase-2,high
ENSG00000132170,PPARG,Lipid,metabolic,Peroxisome proliferator-activated receptor gamma,high
ENSG00000072310,SREBF1,Lipid,metabolic,Sterol regulatory element-binding protein 1,medium
ENSG00000175445,LPL,Lipid,metabolic,Lipoprotein lipase,high
ENSG00000170323,FABP4,Lipid,metabolic,Fatty acid-binding protein 4,medium
ENSG00000174697,LEP,Adipokine,metabolic,Leptin,high
ENSG00000181092,ADIPOQ,Adipokine,metabolic,Adiponectin,high
ENSG00000104918,RETN,Adipokine,metabolic,Resistin,medium
ENSG00000105550,FGF21,Adipokine,metabolic,Fibroblast growth factor 21,high
ENSG00000136869,TLR4,Innate_Immunity,infectious,Toll-like receptor 4,high
ENSG00000172936,MYD88,Innate_Immunity,infectious,Myeloid differentiation primary response protein MyD88,high
ENSG00000186803,IFNA1,Innate_Immunity,infectious,Interferon alpha-1,medium
ENSG00000171855,IFNB1,Innate_Immunity,infectious,Interferon beta,high
ENSG00000167286,CD3D,Adaptive_Immunity,infectious,T-cell surface glycoprotein CD3 delta chain,high
ENSG00000177455,CD19,Adaptive_Immunity,infectious,B-lymphocyte antigen CD19,high
ENSG00000211895,IGHA1,Adaptive_Immunity,infectious,Immunoglobulin heavy constant alpha 1,medium
ENSG00000211896,IGHG1,Adaptive_Immunity,infectious,Immunoglobulin heavy constant gamma 1,high
ENSG00000136244,IL6,Inflammation,infectious,Interleukin-6,high
ENSG00000232810,TNF,Inflammation,infectious,Tumor necrosis factor,high
ENSG00000125538,IL1B,Inflammation,infectious,Interleukin-1 beta,high
ENSG00000169429,CXCL8,Inflammation,infectious,Interleukin-8,medium
ENSG00000164047,CAMP,Antimicrobial,infectious,Cathelicidin antimicrobial peptide,high
ENSG00000164821,DEFB1,Antimicrobial,infectious,Beta-defensin 1,medium
ENSG00000090382,LYZ,Antimicrobial,infectious,Lysozyme C,high
ENSG00000005381,MPO,Antimicrobial,infectious,Myeloperoxidase,high"""
    
    # Create directories
    os.makedirs("../data", exist_ok=True)
    
    # Write gene data to paste.txt for reference
    with open("paste.txt", "w") as f:
        f.write(gene_data)
    
    # Parse data into a DataFrame
    import io
    df = pd.read_csv(io.StringIO(gene_data))
    
    # Add sample_id column
    df["sample_id"] = [f"SAMPLE_{i:03d}" for i in range(len(df))]
    
    # Print dataset info
    print(f"Created gene dataset with {len(df)} samples")
    print(f"Disease types: {', '.join(df['disease_type'].unique())}")
    print(f"Disease distribution:")
    for disease, count in df['disease_type'].value_counts().items():
        print(f"  - {disease}: {count} samples")
    
    # Split into training and testing sets
    train_df, test_df = train_test_split(
        df, 
        test_size=0.25, 
        stratify=df["disease_type"],
        random_state=42
    )
    
    # Save datasets
    train_df.to_csv("../data/Training.csv", index=False)
    print(f"✅ Created Training.csv with {len(train_df)} samples")
    
    test_df.to_csv("../data/Testing.csv", index=False)
    print(f"✅ Created Testing.csv with {len(test_df)} samples")
    
    # Create a version without labels for prediction testing
    test_no_labels = test_df.drop(columns=["disease_type"])
    test_no_labels.to_csv("../data/Testing_no_labels.csv", index=False)
    print(f"✅ Created Testing_no_labels.csv for prediction testing")
    
    return train_df, test_df

if __name__ == "__main__":
    print("🧬 Preparing gene datasets...")
    train_df, test_df = create_gene_datasets()
    print("\n✅ Dataset preparation complete. You can now run the train_model.py script.")