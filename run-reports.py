import sys
import src.reports.report_item_fail_rates_deployment as deployment
import src.reports.report_item_fail_rates_pilot as pilot

def main():
    
    if sys.argv[1] == 'deployment' :
        
        deployment.main()
        
    if sys.argv[1] == 'pilot' :
        
        pilot.main()


if __name__ == "__main__":

    main()
