import os
import json


import pandas as pd




# diversity_model_label_mapping = {
#     0: "background",
#     1: "purpose",
#     2: "method", 
#     3: "finding",
#     4: "other",
# }


def main():
    acl_df = pd.read_csv("./ACL.csv")
    chi_df = pd.read_csv("./CHI.csv")
    iclr_df = pd.read_csv("./ICLR.csv")
    ###### ['id', 'text', 'aspect', 'aspect_confidence', 'perplexity', 'token_count'] ######

    # print(acl_df.head())


    acl_token_count_list = acl_df['token_count']
    print(f"======>>> Max={acl_token_count_list.max()}, Min={acl_token_count_list.min()}, Mean={acl_token_count_list.mean()}")

    chi_token_count_list = chi_df['token_count']
    print(f"======>>> Max={chi_token_count_list.max()}, Min={chi_token_count_list.min()}, Mean={chi_token_count_list.mean()}")

    iclr_token_count_list = iclr_df['token_count']
    print(f"======>>> Max={iclr_token_count_list.max()}, Min={iclr_token_count_list.min()}, Mean={iclr_token_count_list.mean()}")


    # print("aspect:", acl_df['aspect'])
    # print("chi_df aspect:", chi_df['aspect'])
    # print("iclr_df aspect:", iclr_df['aspect'])
    # print(acl_df.to_string()) 


### Debug
if __name__ == '__main__':
    main()




