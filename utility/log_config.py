"""
Used to configure the log for each dataset in preprocessing/log_to_history.py
[2025-05-09]: added 'orbassano' dataset
"""
log = {
        'helpdesk': {'event_attribute': ['activity', 'resource', 'timesincecasestart','servicelevel','servicetype','workgroup','product','customer'], 'trace_attribute': ['supportsection','responsiblesection'],
                     'event_template': '{{resource}} performed {{activity}} {{timesincecasestart}} seconds ago from the beginning of the trace. {{workgroup}} managed the request for the {{product}} of {{customer}} with service {{servicetype}} of level {{servicelevel}}.',
                     'trace_template': 'Section {{supportsection}} led by {{responsiblesection}}', 'target':'activity'},


        'sepsis': {'event_attribute': ['activity','orggroup','timesincecasestart', 'Leucocytes','CRP','LacticAcid'], 'trace_attribute': ['InfectionSuspected','DiagnosticBlood','DisfuncOrg','SIRSCritTachypnea','Hypotensie','SIRSCritHeartRate','Infusion','DiagnosticArtAstrup','Age','DiagnosticIC','DiagnosticSputum','DiagnosticLiquor','DiagnosticOther','SIRSCriteria2OrMore','DiagnosticXthorax','SIRSCritTemperature','DiagnosticUrinaryCulture','SIRSCritLeucos','Oligurie','DiagnosticLacticAcid','Diagnose','Hypoxie','DiagnosticUrinarySediment','DiagnosticECG'],
                   'event_template': 'Org{{orggroup}} performed {{activity}} {{timesincecasestart}} seconds ago from the beginning of the trace. Leucocytes {{Leucocytes}} CRP {{CRP}} LacticAcid {{LacticAcid}}.',
                   'trace_template': 'Patient with Age {{Age}} clinic status: InfectionSuspected {{InfectionSuspected}} DiagnosticBlood {{DiagnosticBlood}} DisfuncOrg {{DisfuncOrg}} SIRSCritTachypnea {{SIRSCritTachypnea}} Hypotensie {{Hypotensie}} SIRSCritHeartRate {{SIRSCritHeartRate}} Infusion {{Infusion}} DiagnosticArtAstrup {{DiagnosticArtAstrup}} DiagnosticIC {{DiagnosticIC}} DiagnosticSputum {{DiagnosticSputum}} DiagnosticLiquor {{DiagnosticLiquor}} DiagnosticOther {{DiagnosticOther}} SIRSCriteria2OrMore {{SIRSCriteria2OrMore}} DiagnosticXthorax {{DiagnosticXthorax}} SIRSCritTemperature {{SIRSCritTemperature}} DiagnosticUrinaryCulture {{DiagnosticUrinaryCulture}} SIRSCritLeucos {{SIRSCritLeucos}} Oligurie {{Oligurie}} DiagnosticLacticAcid {{DiagnosticLacticAcid}} Diagnose {{Diagnose}} Hypoxie {{Hypoxie}} DiagnosticUrinarySediment {{DiagnosticUrinarySediment}} DiagnosticECG {{DiagnosticECG}}.', 'target':'activity'},

        'bpic2020': {'event_attribute': ['activity','resource','timesincecasestart', 'Role'], 'trace_attribute': ['Org','Project','Task'],
                     'event_template': '{{resource}} with role {{Role}} performed {{activity}} {{timesincecasestart}} seconds ago from the beginning of the trace.',
                     'trace_template': '{{Org}} managed the {{Project}} for {{Task}}.', 'target':'activity'},

        'BPIC15_1': {'event_attribute': ['activity', 'resource','timesincecasestart', 'question', 'monitoringResource'], 'trace_attribute': ['parts', 'responsibleactor', 'lastphase', 'landregisterid', 'casestatus', 'sumleges'],
                     'event_template': 'Res{{resource}} performed {{activity}} {{timesincecasestart}} seconds ago from the beginning of the trace. The open question {{question}} concerned {{monitoringResource}}.',
                     'trace_template': 'The application concerned the status {{casestatus}} of the {{parts}} as part of {{lastphase}} in the project associated with LandRegisterID: {{landregisterid}} with {{sumleges}} and responsible {{responsibleactor}}.', 'target':'activity'},

        'bpic2017_o': {'event_attribute': ['activity', 'resource', 'timesincecasestart', 'action'], 'trace_attribute': ["MonthlyCost", "CreditScore", "FirstWithdrawalAmount", "OfferedAmount","NumberOfTerms"],
                       'event_template': '{{resource}} performed {{activity}} with status {{action}} {{timesincecasestart}} seconds ago from the beginning of the trace.',
                       'trace_template': 'The MonthlyCost {{MonthlyCost}} for the loan, determined based on the score {{CreditScore}}, calculated considering the FirstWithdrawalAmount {{FirstWithdrawalAmount}}, the OfferedAmount {{OfferedAmount}}, and the NumberOfTerms {{NumberOfTerms}}.','target': 'activity'},

        'mip': {'event_attribute': ['activity','resource','timesincecasestart','numsession','userid','turn','userutterance','chatbotresponse'], 'trace_attribute': [],
                'event_template': '{{resource}} performed {{activity}} {{timesincecasestart}} seconds ago from the beginning of the trace. In session {{numsession}} turn {{turn}} the user utterance was {{userutterance}} and chatbot response was {{chatbotresponse}}.',
                'trace_template': '', 'target':'activity'},

        'prova_OUTPUT': {'event_attribute': ['activity','resource','timestamp','CURRENT_ESI_1','CURRENT_ESI_2','CURRENT_ESI_3','CURRENT_ESI_4','CURRENT_ESI_5'], 'trace_attribute': ['ESI', 'OUTCOME', 'INPAT_HOSP_DEP'],
                         'event_template': '{{resource}} started performing {{activity}} at {{timestamp}} on a patient. At the moment of the activity, {{CURRENT_ESI_1}} patients had a ESI of level 1, {{CURRENT_ESI_2}} patients had a ESI of level 2, {{CURRENT_ESI_3}} patients had a ESI of level 3, {{CURRENT_ESI_4}} patients had a ESI of level 4, {{CURRENT_ESI_5}} patients had a ESI of level 5.',
                         'trace_template': 'The patient was assigned with ESI {{ESI}}. The outcome for the patient is {{OUTCOME}} in hospital department {{INPAT_HOSP_DEP}}.', 'target':'activity'},
        
        'orbassano': {'event_attribute': ['activity','resource','timestamp','CURRENT_ESI_1','CURRENT_ESI_2','CURRENT_ESI_3','CURRENT_ESI_4','CURRENT_ESI_5'], 'trace_attribute': ['ESI', 'OUTCOME', 'INPAT_HOSP_DEP'],
                         'event_template': '{{resource}} started performing {{activity}} at {{timestamp}} on a patient. At the moment of the activity, {{CURRENT_ESI_1}} patients had a ESI of level 1, {{CURRENT_ESI_2}} patients had a ESI of level 2, {{CURRENT_ESI_3}} patients had a ESI of level 3, {{CURRENT_ESI_4}} patients had a ESI of level 4, {{CURRENT_ESI_5}} patients had a ESI of level 5.',
                         'trace_template': 'The patient was assigned with ESI {{ESI}}. The outcome for the patient is {{OUTCOME}} in hospital department {{INPAT_HOSP_DEP}}.', 'target':'activity'}

}



