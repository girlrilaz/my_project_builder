import sys
import src.reports.report_item_fail_rates_deployment as deployment

def main():
    
    if sys.argv[1] == 'deployment' :
        
        deployment.main()
        
    if sys.argv[1] == 'pilot' :
        
        print('Pilot RUN!')


if __name__ == "__main__":

    main()
