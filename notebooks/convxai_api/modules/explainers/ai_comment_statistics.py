global_explanations_data = {
            "ACL": {
                "paper_count": 3221,
                "sentence_count": 20744,
                "sentence_length": 
                    {
                        "all":        [2, 14, 26, 38, 50],
                        "background": [4, 14, 23, 33, 42],
                        "purpose":    [6, 16, 27, 37, 47],
                        "method":     [0, 14, 27, 41, 55],
                        "finding":    [1, 14, 26, 39, 52],
                        "other":      [-11,-3,5,  14, 22]
                    },
                "sentence_score_range": 
                    {
                        "all": [22, 32, 39, 46, 71],
                        "background": [21, 31, 36, 44, 68],
                        "purpose": [19, 26, 30, 35, 51],
                        "method": [27, 38, 44, 53, 78],
                        "finding": [22, 33, 40, 48, 76],
                        "other": [36, 63, 102, 213, 692]
                    },
                "abstract_score_range": [32, 41, 46, 51, 67],
                "aspect_distribution": [],
                "Aspect_Patterns_dict": {
                    "00122233": "'background' (25%)   -&gt; 'purpose' (12.5%) -&gt; 'method'  (37.5%) -&gt; 'finding' (25%)",
                    "001233": "'background' (33.3%) -&gt; 'purpose' (16.7%) -&gt; 'method'  (16.7%) -&gt; 'finding' (33.3%)",
                    "0002233": "'background' (42.9%) -&gt; 'method'  (28.6%) -&gt; 'finding' (28.5%)",
                    "000133": "'background' (50%)   -&gt; 'purpose' (16.7%) -&gt; 'finding' (33.3%)",
                    "00323333": "'background' (25%)   -&gt; 'finding' (12.5%) -&gt; 'method'  (12.5%) -&gt; 'finding' (50%)",
                },
            },
            "CHI": {
                "paper_count": 3235,
                "sentence_count": 21643,
                "sentence_length": 
                    {
                        "all":        [4, 14, 25, 36, 46],
                        "background": [5, 14, 22, 31, 40],
                        "purpose":    [6, 17, 27, 38, 49],
                        "method":     [4, 15, 27, 39, 51],
                        "finding":    [4, 15, 26, 37, 48],
                        "other":      [-3,0,  4,  7,  11]
                    },
                "sentence_score_range": 
                    {
                        "all": [32, 45, 53, 63, 97],
                        "background": [28, 40, 48, 57, 88],
                        "purpose": [28, 39, 45, 52, 75],
                        "method": [33, 47, 56, 67, 103],
                        "finding": [37, 52, 62, 73, 111],
                        "other": [21, 117, 177, 523, 5979]
                    },
                "abstract_score_range": [45, 57,  63, 71, 92],
                "aspect_distribution": [],
                "Aspect_Patterns_dict": {
                    "0001333": "'background' (42.9%) -&gt; 'purpose' (14.3%)  -&gt; 'finding' (42.9%)",
                    "001222333": "'background' (22.2%) -&gt; 'purpose' (11.2%) -&gt; 'method' (33.3%) -&gt; 'finding' (33.3%)",
                    "001233": "'background' (33.3%) -&gt; 'purpose' (16.7%) -&gt; 'method' (16.7%)  -&gt; 'finding' (33.3%)",
                    "002333": "'background' (33.3%) -&gt; 'method' (16.7%)  -&gt;  'finding' (50%)",
                    "000300100323333": "'background' (20%)   -&gt; 'finding' (6.7%)  -&gt;  'background' (13.3%) -&gt; 'purpose' (6.7%) -&gt; 'background' (13.3%) -&gt; 'finding' (6.7%) -&gt; 'method' (6.7%) -&gt; 'finding' (26.7%)"
                },
            },
            "ICLR": {
                "paper_count": 3479,
                "sentence_count": 25873,
                "sentence_length": 
                    {
                        "all":        [0, 13, 27, 41, 54],
                        "background": [2, 13, 24, 36, 47],
                        "purpose":    [4, 16, 28, 39, 51],
                        "method":     [-2,13, 29, 45, 61],
                        "finding":    [0, 14, 28, 42, 55],
                        "other":      [-1,2,  5,  8,  11]
                    },
                "sentence_score_range": 
                    {
                        "all": [35, 52, 62, 74, 116],
                        "background": [32, 48, 57, 68, 106],
                        "purpose": [28, 40, 47, 56, 83],
                        "method": [40, 58, 68, 82, 125],
                        "finding": [37, 56, 66, 80, 126],
                        "other": [31, 50, 59, 79, 371]
                    },
                "abstract_score_range": [52, 67,  76, 84, 111],
                "aspect_distribution": [],
                "Aspect_Patterns_dict": {
                    "001233": "'background' (33.3%) -&gt; 'purpose' (16.7%) -&gt; 'method' (16.7%) -&gt; 'finding' (33.3%)",
                    "23333": "'Method' (20%) -&gt; 'finding' (80%)",
                    "0001333": "'background' (42.9%) -&gt; 'purpose' (14.2) -&gt; 'finding' (42.9%)",
                    "00000232333": "'background' (45.5%) -&gt; 'method' (9.1%) -&gt; 'finding' (9.1%) -&gt; 'method' (9.1%) -&gt; 'finding' (27.3%)",
                    "001222333": "'Background' (22.2%) -&gt; 'purpose' (11.1%) -&gt; 'method' (33.3%) -&gt; 'finding' (33.4%)",
                }
            },
            "Aspect_Patterns-ACL":[
                "'background' (25%)   -&gt; 'purpose' (12.5%) -&gt; 'method'  (37.5%) -&gt; 'finding' (25%)",
                "'background' (33.3%) -&gt; 'purpose' (16.7%) -&gt; 'method'  (16.7%) -&gt; 'finding' (33.3%)",
                "'background' (42.9%) -&gt; 'method'  (28.6%) -&gt; 'finding' (28.5%)",
                "'background' (50%)   -&gt; 'purpose' (16.7%) -&gt; 'finding' (33.3%)",
                "'background' (25%)   -&gt; 'finding' (12.5%) -&gt; 'method'  (12.5%) -&gt; 'finding' (50%)",
            ],
            "Aspect_Patterns-CHI":[
                "'background' (42.9%) -&gt; 'purpose' (14.3%)  -&gt; 'finding' (42.9%)",
                "'background' (22.2%) -&gt; 'purpose' (11.2%) -&gt; 'method' (33.3%) -&gt; 'finding' (33.3%)",
                "'background' (33.3%) -&gt; 'purpose' (16.7%) -&gt; 'method' (16.7%)  -&gt; 'finding' (33.3%)",
                "'background' (33.3%) -&gt; 'method' (16.7%)  -&gt;  'finding' (50%)",
                "'background' (20%)   -&gt; 'finding' (6.7%)  -&gt;  'background' (13.3%) -&gt; 'purpose' (6.7%) -&gt; 'background' (13.3%) -&gt; 'finding' (6.7%) -&gt; 'method' (6.7%) -&gt; 'finding' (26.7%)"
            ],
            "Aspect_Patterns-ICLR":[
                "'background' (33.3%) -&gt; 'purpose' (16.7%) -&gt; 'method' (16.7%) -&gt; 'finding' (33.3%)",
                "'Method' (20%) -&gt; 'finding' (80%)",
                "'background' (42.9%) -&gt; 'purpose' (14.2) -&gt; 'finding' (42.9%)",
                "'background' (45.5%) -&gt; 'method' (9.1%) -&gt; 'finding' (9.1%) -&gt; 'method' (9.1%) -&gt; 'finding' (27.3%)",
                "'Background' (22.2%) -&gt; 'purpose' (11.1%) -&gt; 'method' (33.3%) -&gt; 'finding' (33.4%)",
            ]
        }


