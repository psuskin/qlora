from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
    Seq2SeqTrainer,
    BitsAndBytesConfig,
    LlamaTokenizer

)

new_toks = ['about', 'accarea', 'access', 'acntstat', 'address', 'addressFormats', 'addrtyp', 'aditmsel', 'advancedSearch', 'advatsta', 'airplane', 'airport', 'alarmclk', 'alert', 'alloccon', 'allocsng', 'alocitem', 'alocwork', 'analyitd', 'analypdc', 'appCopierEdit', 'appGeneratorEdit', 'appInheritorEdit', 'approvalTransactions', 'appsched', 'appsWHBrowser', 'appsWHModuleSelect', 'appsWHModuleSynchronise', 'assetAccountBalanceList', 'assetAccountTxnList', 'assetsAnalysisList', 'associatl', 'atsetobj', 'attrcbag', 'attrform', 'attribut', 'attributeValueEdit', 'attrilst', 'attrisat', 'attrisit', 'attrnode', 'attrslot', 'auditing', 'autopcal', 'autoplst', 'aveprice', 'balanbus', 'balances', 'balancos', 'balanfac', 'bank', 'bankaedt', 'bankcode', 'batcerr', 'bbook', 'billcond', 'billing', 'billofma', 'billsing', 'birthday', 'blockers', 'body-mod002', 'body-mod003', 'body-modgenoview', 'body-tools', 'branches', 'budofedt', 'busiseg', 'busiyear', 'calendar', 'canceltxn', 'capacityPlanning', 'capcheck', 'car', 'carbonco', 'CarPrio', 'cashDeposit', 'cashdisc', 'ccacbals', 'ccacbstr', 'chargbas', 'cheqregi', 'classificl', 'clipboard', 'clsstree', 'cmacbals', 'coacbals', 'coacstat', 'cobjrept', 'columvar', 'comment', 'commiss', 'condaccn', 'condcond', 'condiset', 'condtedc', 'condtedt', 'condtion', 'consult', 'conversation', 'coobwizz', 'corporat', 'corporateGroupEdit', 'costbook', 'costcent', 'costiobj', 'costmobj', 'costObjectiveBalanceList', 'costosel', 'costpobj', 'costsobj', 'costtype', 'cp_1252', 'cracbals', 'csearch', 'cshowsit', 'ctacbals', 'ctacbstr', 'curracc', 'currencyExchange', 'cusgroup', 'customagr', 'customer', 'cxAccessNode', 'cxAccessoryList', 'cxAdvanceDemand', 'cxApiKey', 'cxApplication', 'cxAsset', 'cxCampaign', 'cxCity', 'cxCombinedNomenclature', 'cxConditionedBag', 'cxContact', 'cxCounter', 'cxCreditCardAccount', 'cxCurrencyTable', 'cxCustomerSurvey', 'cxCyberClient', 'cxDatabase', 'cxDataConnector', 'cxDataField', 'cxDocument', 'cxDocumentComponent', 'cxDocumentIndex', 'cxFacility', 'cxForkTruck', 'cxGeneralLedger', 'cxIndustrialPlant', 'cxInstallationCharges', 'cxIntermediateStocking', 'cxInvestmentBudget', 'cxIpAddress', 'cxItemCostEstimate', 'cxItemDemand', 'cxIvWidgetComponent', 'cxModuleSettings', 'cxNeuralNetwork', 'cxNeuralNetwork2', 'cxOutput', 'cxPattItemNumber', 'cxPhrase', 'cxPickList', 'cxPotentialPartner', 'cxProceedings', 'cxProductionDataCapture', 'cxPurchaseReturn', 'cxReport', 'cxSalesProductConfiguration', 'cxSapBusinessOneStock', 'cxSignification', 'cxStatement', 'cxStateMonitor', 'cxStateStructure', 'cxStockExecutionItem', 'cxStockSpace', 'cxStructure', 'cxTextCondition', 'cxTxnDescriptor', 'cxTxnNote', 'cxWebService', 'cxWidget', 'cxWorkflow', 'cxWorkFlowRoute', 'cxWorkTimeEvent', 'cxWorkTimeModel', 'cxWorkTimeRule', 'cxWorkTimeYear', 'cyber', 'databaseManage', 'dataConnectorImport', 'dataConnectorWebBrowser', 'dataFieldPathSelect', 'date', 'dbaseviw', 'deacbals', 'defclass', 'defslot', 'deftrans', 'deletion', 'deliauto', 'deliconf', 'delidisp', 'deliisel', 'delinote', 'deliveryNoteItemList', 'delroute', 'deveitem', 'dialogue', 'dictview', 'directShipmentItem', 'dirshipm', 'discount', 'dispobom', 'dnpycus', 'dnpysup', 'domadecl', 'dprcbook', 'dtausedt', 'dtazvedt', 'dunnsele', 'ecvatsta', 'eMailSentStatus', 'enumerat', 'exacbals', 'excelcel', 'excelReader', 'eximport', 'ExpandNum', 'exprecei', 'extdispo', 'family', 'favourit', 'fiacstat', 'finabook', 'finacopy', 'finajour', 'financialLoggingsList', 'finpcacc', 'finstand', 'flextime', 'flightOperation', 'floatfil', 'flowChart', 'forecast', 'formsamp', 'formula', 'forwardr', 'fwdfabal', 'gantitdm', 'gantosup', 'gantt', 'gantwork', 'geledcla', 'geleddep', 'generalLedgerBalancesStructureList', 'genmodul', 'genproce', 'glacbals', 'globvar', 'gozinto', 'graphicalQueryWizard', 'hbaytxn', 'hbstock', 'helpgen', 'holdinre', 'icastedt', 'ImportStockSpace', 'indexmgr', 'initsdat', 'inprovis', 'inspection', 'instchbook', 'intrastat', 'invanaly', 'invcontr', 'inventoryAnalysis', 'inventoryCheck', 'inventoryCurrentList', 'inventoryFrequencyDistribution', 'inventoryImport', 'inventoryStatistics', 'inventoryStratification', 'inventoryTaking', 'inventry', 'invoice', 'invoiitm', 'invoimaint', 'isearch', 'item', 'itemAlloc', 'itemDispositionEdit', 'itemDispositionLoggingsSelect', 'itemsea', 'itemVarianceAnalyze', 'jobRecordByDayWin', 'jobrecrd', 'jobsched', 'jobscond', 'jobssing', 'journey', 'keyshtct', 'kpiauditing', 'kpiAuditor', 'kpiMonitor', 'kpisupmo', 'language', 'legalPersonDeduplicate', 'legalPersonDuplicatesList', 'legalPersonNamesList', 'listvcol', 'literalAppsWH', 'literalSystem', 'loadeddlls', 'localeEdit', 'logcyber', 'logcyuse', 'logfistx', 'loggiadm', 'loggibom', 'loggicid', 'loggicos', 'loggidac', 'loggidit', 'loggings', 'loggiocm', 'loggiocr', 'loggiode', 'loggioex', 'loggioit', 'loggipic', 'loggipit', 'loggipoi', 'loggipor', 'loggiprov', 'loggirit', 'loggisto', 'loggiwfl', 'loggiwip', 'loggiwst', 'login', 'loginAgent', 'loginAlias', 'logindex', 'logout', 'loguser', 'machine', 'mail', 'maintenc', 'maintpaym', 'manufact', 'masttool', 'member', 'message', 'messaint', 'metaaccs', 'metainfo', 'metamodl', 'metaobj', 'miniwb', 'missingAttributes', 'modgen', 'modoview', 'moniinsp', 'montagestuecklisten', 'motolist', 'mt940edt', 'multiLingual', 'neuralNetworkLoad', 'neuralNetworkQuery', 'newsletter2Go', 'objcount', 'objctedt', 'objctlog', 'objctsql', 'objectChangeLogList', 'objectsLoad', 'objectStructureEdit', 'objectWebCrawler', 'objinsp', 'objnavi', 'objterms', 'odbcTestSuite', 'odocu', 'offorcal', 'offorctr', 'offorder', 'offorhqu', 'offorita', 'offoritt', 'offorprn', 'ofitmsel', 'oitemsel', 'oitmsupp', 'olapCustomerList', 'olapRepresentativeList', 'olapSupplierList', 'olsync', 'olsyncmore', 'onewaypa', 'openitem', 'openItemTxnSelect', 'openlist', 'operator', 'opitcmac', 'opitcrac', 'opitdbac', 'opitdunn', 'opitexac', 'opitglac', 'orderAllocationResolve', 'orderctr', 'orderfav', 'ordergrp', 'ordermaint', 'orderPaymentStatisticsList', 'orderqui', 'orderrisk', 'orderStateStatisticsList', 'ordertxt', 'organizationChart', 'origpart', 'ortodnin', 'oslotbas', 'outlook', 'outprovis', 'packaccn', 'packitem', 'packload', 'parquery', 'partner', 'partnerCastEdit', 'partnerpflege', 'partpaym', 'password', 'paychequ', 'paydtaus', 'paydtazv', 'payevdnc', 'payments', 'paymprop', 'paymt101', 'paysepa', 'pcalcgrp', 'perdate', 'performa', 'person', 'personDeduplicate', 'personDuplicatesList', 'personNamesList', 'picture', 'pinvbook', 'plusbutton', 'pmatchcode', 'pmedia', 'ppcrbals', 'ppdebals', 'presentationManagerEdit', 'preview', 'pricecal', 'PriceDiscount', 'PriceDiscountTable', 'PriceExamples', 'prichap', 'prichas', 'print', 'printer', 'printIndex', 'printole', 'printProblems', 'printtyp', 'prjgen', 'prjnmoni', 'processes_abschluss00', 'processes_abschluss01', 'processes_abschluss02', 'processes_accountancy', 'processes_accounting', 'processes_addresses', 'processes_advancedemand', 'processes_applidev', 'processes_attribute', 'processes_bpordercalculation', 'processes_bprocextjobsched', 'processes_bproclog', 'processes_bprocprod', 'processes_bprocpurch', 'processes_bprocret', 'processes_bprocsales', 'processes_bprocservice', 'processes_bproitmcest', 'processes_bpsalesbilling', 'processes_bpsalescrm', 'processes_bpsalesdelinote', 'processes_calculation', 'processes_capsched', 'processes_casts', 'processes_changecancelsalesitem', 'processes_changemanag', 'processes_changesalesitemItem', 'processes_changesalesitemquantity', 'processes_chargen', 'processes_Companies_across_processes', 'processes_construction', 'processes_customerauditing', 'processes_deliveryauditing', 'processes_dispatch', 'processes_docind_main', 'processes_docind_purchase', 'processes_documentEditing', 'processes_docvar', 'processes_excel', 'processes_foreigncurrency', 'processes_humanresources', 'processes_inprovisio', 'processes_international', 'processes_item_lifecycle', 'processes_itemzeroquantityauditing', 'processes_kanban', 'processes_keydata', 'processes_lieferantenerklaerung', 'processes_localeReference', 'processes_logistics', 'processes_objaccess', 'processes_officeXML', 'processes_outprovisio', 'processes_phonemail', 'processes_proctype', 'processes_proddatacapt', 'processes_projectlifecycle', 'processes_projects', 'processes_qm_bat', 'processes_qm_fin', 'processes_qm_main', 'processes_qm_mat', 'processes_qm_par', 'processes_qm_sal', 'processes_rdadd', 'processes_rdbo', 'processes_rdcasts', 'processes_rdreg', 'processes_receiving', 'processes_ReengineeringBOM', 'processes_regenrep', 'processes_reportfkt', 'processes_reporting', 'processes_rootdata', 'processes_sales_main', 'processes_sales_masterdata', 'processes_sales_offer', 'processes_sales_order', 'processes_salespricing', 'processes_serinumber', 'processes_stockaccounting', 'processes_supplierauditing', 'processes_supply', 'processes_textitem', 'processes_txndata', 'processes_unitarithm', 'processes_variant01', 'processes_variant02', 'processes_variant03', 'processes_variant04', 'processes_versioning', 'processes_vocabulary', 'procfold', 'prodserv', 'product', 'Profiling', 'ProfilingCluster', 'projaloc', 'projcost', 'projectGeneratorEdit', 'projinfo', 'proorder', 'proquery', 'provsing', 'pstgroup', 'publisher', 'purcappr', 'purccomp', 'purcdunn', 'PurchaseApproval', 'purchaseInquiry', 'purchaseItem', 'purchaseOrder', 'purchaseOrderSignature', 'purchaseProposalsList', 'purchaseRequisition', 'purchaseRequisitionLoggingsList', 'purchaseService', 'purchaseSet', 'purchcre', 'purchinv', 'purchinvitem', 'purchinvlog', 'purciprn', 'puriauto', 'puridunn', 'puriitem', 'purogedt', 'puroitem', 'purqitem', 'pusaitem', 'Pythia_cxAntiTerrorScreening', 'Pythia_legalPersonList_host', 'Pythia_legalPersonList_pythia', 'Pythia_outlook_py', 'Pythia_personList_host', 'Pythia_personList_pythia', 'Pythia_sanctionsListMatch', 'Pythia_sanctionsListMonitor', 'Pythia_sanctionsListQuery', 'Pythia_xmlimprt_py', 'qm_alert_qm', 'qm_arithmetic_qm', 'qm_asciiFile_qm', 'qm_button_qm', 'qm_commap_qm', 'qm_condBag_qm', 'qm_condWrapper_qm', 'qm_date_qm', 'qm_deadlock_qm', 'qm_dictionary_qm', 'qm_dragndrop_qm', 'qm_draw_qm', 'qm_font_qm', 'qm_formula_qm', 'qm_garbage_qm', 'qm_grpwid_qm', 'qm_helpinfo_qm', 'qm_itempattern_qm', 'qm_link_qm', 'qm_listview_qm', 'qm_listviewAutoPos_qm', 'qm_listviewExceptions_qm', 'qm_listviewOboxEdit_qm', 'qm_listviewOboxUpDown_qm', 'qm_listviewOboxUpDown2_qm', 'qm_listviewSetFormat_qm', 'qm_listviewSort_qm', 'qm_listviewxml_qm', 'qm_message_qm', 'qm_mlole_qm', 'qm_olectl_qm', 'qm_patternQuery_qm', 'qm_periodicDate_qm', 'qm_phone_qm', 'qm_picture_qm', 'qm_pobox_qm', 'qm_printing_qm', 'qm_qm', 'qm_query_qm', 'qm_queryExist_qm', 'qm_rates_qm', 'qm_refbasedptr_qm', 'qm_relations_qm', 'qm_reload_qm', 'qm_remoteMsg_qm', 'qm_resume_qm', 'qm_rounding_qm', 'qm_security_qm', 'qm_setLocale_qm', 'qm_simplwid_qm', 'qm_spanDate_qm', 'qm_spanTime_qm', 'qm_systemObject_qm', 'qm_telephony_qm', 'qm_term_qm', 'qm_time_qm', 'qm_timedTrigger_qm', 'qm_tmprture_qm', 'qm_txnByCond_qm', 'qm_unit_qm', 'qm_unittestDLL_qm', 'qm_unittestIV_qm', 'qm_vector_qm', 'qm_wrapper_qm', 'qmbase', 'qs-all', 'qs-purchaseinvoice', 'qs-validity', 'qualassu', 'query', 'queryatt', 'queryManager', 'queryWizard', 'receitem', 'receivingItemStatusList', 'receving', 'registerStorageAids', 'remotmsg', 'rentitem', 'repclass', 'reporting', 'represen', 'request', 'resolbom', 'resoljob', 'resolved', 'resscond', 'resset', 'restrict', 'routePlanningEdit', 'sacoest', 'salebase', 'salecedt', 'salecond', 'saleiprn', 'saleitem', 'saleserv', 'saleset', 'salesinvitemsel', 'salesItemOrderStatisticsList', 'salesman', 'salesOrderEngineering', 'saletend', 'salexitm', 'sampqedt', 'saordrep', 'sapBusinessOneInterfaceMonitor', 'sarestat', 'saretour', 'scanner_dialog_error_app_scanner', 'scanner_dialog_yesnodialog_app_scanner', 'scanner_login_app_scanner', 'scanner_main_app_scanner', 'scanner_main_info_iteminfo_app_scanner', 'scanner_main_info_queryitem_app_scanner', 'scanner_main_info_querystorage_app_scanner', 'scanner_main_info_storageinfo_app_scanner', 'scanner_main_maintenance_adjustinventory_adjustinventorydown_app_scanner', 'scanner_main_maintenance_adjustinventory_adjustinventoryup_app_scanner', 'scanner_main_maintenance_adjustinventory_app_scanner', 'scanner_main_maintenance_adjustinventory_changestatus_app_scanner', 'scanner_main_maintenance_relocate_app_scanner', 'scanner_main_maintenance_relocate_findstoragesofitem_app_scanner', 'scanner_main_maintenance_relocate_license_app_scanner', 'scanner_main_maintenance_relocate_looseitems_app_scanner', 'scanner_main_maintenance_relocate_standardcontainer_app_scanner', 'scanner_main_maintenance_stocktaking_app_scanner', 'scanner_main_maintenance_stocktaking_cyclecountstorage_app_scanner', 'scanner_main_maintenance_stocktaking_inventory_app_scanner', 'scanner_main_maintenance_stocktaking_snrcountstorage_app_scanner', 'scanner_main_print_enterparameters_app_scanner', 'scanner_main_print_labelorreport_app_scanner', 'scanner_main_print_printdocument_app_scanner', 'scanner_main_print_printdocument_printdeliverynote_app_scanner', 'scanner_main_print_printdocument_printdeliverynotefrompicklist_app_scanner', 'scanner_main_print_printdocument_printinvoice_app_scanner', 'scanner_main_print_printitemlabel_app_scanner', 'scanner_main_print_printstoragelabel_app_scanner', 'scanner_main_processes_inbound_app_scanner', 'scanner_main_processes_inbound_checkin_app_scanner', 'scanner_main_processes_inbound_directputaway_app_scanner', 'scanner_main_processes_inbound_receivefromcustomer_app_scanner', 'scanner_main_processes_inbound_receivefromsupplier_app_scanner', 'scanner_main_processes_outbound_app_scanner', 'scanner_main_processes_outbound_pack_app_scanner', 'scanner_main_processes_outbound_pick_app_scanner', 'scanner_main_processes_outbound_selectpackage_app_scanner', 'scanner_main_processes_outbound_shiptocustomer_app_scanner', 'scanner_main_processes_outbound_shiptocustomer_consolidateandship_app_scanner', 'scanner_main_processes_outbound_shiptocustomer_consolidatelicense_app_scanner', 'scanner_main_processes_outbound_shiptocustomer_consolidatelooseitems_app_scanner', 'scanner_main_processes_outbound_shiptocustomer_identifylicensesofshipment_app_scanner', 'scanner_main_processes_outbound_shiptocustomer_pack_app_scanner', 'scanner_main_processes_outbound_shiptocustomer_pickandcollect_app_scanner', 'scanner_main_processes_outbound_shiptocustomer_pickanddrop_app_scanner', 'scanner_main_processes_transport_app_scanner', 'scanner_main_processes_transport_putaway_app_scanner', 'scanner_main_processes_transport_putaway_fastputaway_nui_app_scanner', 'scanner_main_processes_transport_putaway_licenseputaway_app_scanner', 'scanner_main_processes_transport_putaway_looseitemsputaway_app_scanner', 'scanner_main_processes_transport_putaway_unmixedlicenseputaway_app_scanner', 'scanner_select_blackhole_app_scanner', 'scanner_select_businessdocumentprinter_app_scanner', 'scanner_select_forktruck_app_scanner', 'scanner_select_layout_app_scanner', 'scanner_select_picklist_nui_app_scanner', 'scanner_select_pickzone_app_scanner', 'scanner_select_printer_app_scanner', 'scanner_select_standardcontainer_app_scanner', 'scanner_select_status_app_scanner', 'scanner_select_storage_app_scanner', 'scanner_select_workzone_app_scanner', 'scanner_serialnumbersandbatches_collect_app_scanner', 'scanner_show_consolidationstoragesstatus_app_scanner', 'schedule', 'search', 'secclass', 'secgroup', 'secmessg', 'secobjec', 'secsystm', 'security', 'securityExam', 'segview', 'serienbrief', 'serinumb', 'service', 'serviceTicket', 'serviceTicketClient', 'serviitt', 'servinqu', 'sessiond', 'setalloc', 'setlimit', 'setlocal', 'showmoni', 'showwrkf', 'sinvbook', 'slotsbas', 'soitemsel', 'solutions_pythia_faqs', 'spardire', 'sparitem', 'specifier', 'sstgroup', 'staffmem', 'startset', 'statcust', 'statelink', 'statfoca', 'statinpl', 'statinst', 'statistx', 'statofit', 'statoitm', 'statordr', 'statpodc', 'statprpl', 'statturn', 'statwprg', 'statwrap', 'stoaccnt', 'stock', 'stockInput', 'stockMaterialFlowList', 'stockOrder', 'stockSequentialTest', 'stockSpacePlot', 'stockSpaceQuery', 'StockStatistics', 'stockSwitching', 'stocktxn', 'stockWithdrawal', 'stomobil', 'storage', 'stotrans', 'strmonit', 'submonit', 'substock', 'supgroup', 'supplier', 'supplierAgreement', 'supplierItemList', 'synchrDB', 'sysnote', 'tapi', 'task', 'taxauth', 'taxrate', 'tcpconct', 'telecomEdit', 'telecrep', 'templates_modulvorlage', 'testAllocation', 'testattr', 'testchrg', 'testform', 'timeoffc', 'tool', 'truck', 'txnhisto', 'txnntprn', 'txnuse', 'unitbill', 'unitCalculator', 'units', 'unittabl', 'updFClip', 'user', 'userhier', 'utilaccn', 'utilitem', 'utilofor', 'utilpart', 'utilpurc', 'vacaopen', 'validity', 'vatreturn', 'vehicle', 'verifydb', 'vocabularyGrammer', 'warehouseMonitor', 'webservice', 'windowEdit', 'windows', 'wipAccount', 'workarea', 'workflowGraphList', 'workgrup', 'workingTimeAccount', 'workingTimeAccountReport', 'workrprt', 'workstat', 'workTimeFlexiCalculate', 'workTimeFlexiCloseMonth', 'workTimeFlexiReport', 'workTimeTerminal', 'worldClock', 'z4report', 'ZUGFeRD']

tokenizer = AutoTokenizer.from_pretrained(
    'huggyllama/llama-7b',
    padding_side="right",
    use_fast=False, # Fast tokenizer giving issues.
    tokenizer_type='llama', # Needed for HF name change
)

tokenizer.add_tokens(new_toks)

print(tokenizer.encode("txnuse"))
print(tokenizer.encode("unitbill"))