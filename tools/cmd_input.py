import sys
import argparse

def load_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--dir", 
        type=str,  
        help="Directory with results.", 
        nargs="+")

    parser.add_argument(
        "--problem", 
        choices=["code-gen", "program-repair"],
        default="code-gen",
        help="Which set to evaluate")
    
    parser.add_argument(
        "--all-lang", 
        action="store_true",
        help="Flag to evaluate all languages.")
    
    parser.add_argument(
        "--split", 
        action="store_true",
        help="Flag to split dataset.")  
      
    parser.add_argument(
        "--k-fold", 
        action="store_true",
        default=False,
        help="Use k-fold in IGHB method")   
        
    parser.add_argument(
        "--grouping-style", 
        choices=["simple", "scc", "categories", "all"],
        default="simple",
        help="Choose ways of grouping the samples.")       

    parser.add_argument(
        "--save-table", 
        action="store_true",
        help="Flag to save the result table.")      
    
    parser.add_argument(
        "--prob-method", 
        choices=["avg_logprob", "qualitativ", "quantitativ"],
        default="avg_logprob",
        help="Choose which probability to use.")      

    parser.add_argument(
        "--binning-type", 
        choices=["linear"],
        default='linear',
        help="Choose which binning type to use.")    
    
    parser.add_argument(
        "--bin-count", 
        type=int,  
        default=20,
        help="Choose which binning type to use.")  

    parser.add_argument(
        "--control-exp", 
        action="store_true",
        default=False,
        help="Only for the LR mehtod.")      
    
    parser.add_argument(
        "--save-history", 
        action="store_true",
        default=False,
        help="Only for the IGHB/IGLB method.")  
     
    parser.add_argument(
        "--save-charts", 
        action="store_true",
        default=False,
        help="Save the charts for the method.")  
    
    parser.add_argument(
        "--save-data", 
        action="store_true",
        default=False,
        help="Save the output data of all methods")     
            
    parser.add_argument(
        "--epsilon", 
        type=float,  
        default=0.01,
        help="Epsilon value only for IGLB method.") 
    
    parser.add_argument(
        "--model", 
        choices=["gpt_4o_mini"],
        required=("quantitativ" in sys.argv or "qualitativ" in sys.argv),
        help="Model with which the verbalized data was created") 
         
    args = parser.parse_args()

    return args
