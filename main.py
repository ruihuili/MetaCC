from utils.args_parser import get_args

if __name__ == '__main__':
    args, device = get_args()

    if args.meta_learner.lower() in ["maml", "maml-fo", "metasgd", "metacurvature", "metacurv"]:
        from learners.maml_coding import main
    elif args.meta_learner.lower() in ["erm"]:
        from learners.erm_coding import main
    elif args.meta_learner.lower() in ["reptile"]:
        from learners.reptile_coding import main
    elif args.meta_learner.lower() in ["anil"]:
        from learners.anil_coding import main
    elif args.meta_learner.lower() in ["kfo"]:
        from learners.kfo_coding import main
    elif args.meta_learner.lower() in ["boil"]:
        from learners.boil_coding import main

    elif args.meta_learner.lower() in ["cavia"]:
        from learners.cavia_coding import main

    elif args.meta_learner.lower() in ["protonets", "proto", "proton"]:
        from learners.protonets_coding import main
    elif args.meta_learner.lower() in ["metabaseline"]:
        from learners.metabaseline_long_coding import main

    elif args.meta_learner.lower() in ["viterbi"]:
        from learners.viterbi import main


    main(args, device)
