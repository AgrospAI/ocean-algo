KEYWORD_RULES = {
    "software": {
        "A3 / Wolters Kluwer": ["a3", "wolter", "kluwer", "a3erp"],
        "Sage": ["sage", "murano", "sage50", "sage200", "eurowin"],
        "Microsoft Dynamics": ["navision", "nav", "business central", "dynamics", "bc"],
        "Microsoft Excel": ["excel", "sheet", "hoja", "calculo", "office"],
        "SAP": ["sap", "b1", "business one", "hana"],
        "Odoo": ["odoo", "openerp"],
        "Salesforce": ["salesforce", "force.com"],
        "HubSpot": ["hubspot"],
        "Power BI": ["power", "powerbi", "srss"],
        "A3ERP BI": ["bi", "a3 bi"],
        "Ninguno": ["no"]
    },

    "sector": {
        "Industrial": ["fabric", "manufactur", "taller", "construc", "obra", "instalad", "metal"],
        "Comercio": ["venta", "shop", "tienda", "retail", "comerc", "distribuc", "almacen"],
        "Servicios": ["consult", "asesor", "gestor", "abogad", "software", "marketing", "educacion", "clinic", "servicios"],
        "Tecnología": ["desarrollo", "programacion", "sistemas", "tic", "ciber"]
    },

    "channel": {
        "WhatsApp": ["wha", "was", "wats", "zap"],
        "Email": ["mail", "correo", "outlook", "gmail", "hotmail", "electrónico", "corre"],
        "Teléfono": ["tlf", "cel", "movil", "call", "phone", "fijo", "telefono"],
        "Videollamada": ["teams", "zoom", "meet", "skype"]
    },
    
    "antivirus": {
        "Microsoft Defender": ["defender", "windows", "microsoft", "no usan", "ninguno"],
        "Kaspersky": ["kasp"],
        "McAfee": ["mcafee", "trellix"],
        "Norton/Symantec": ["norton", "symantec"],
        "Bitdefender": ["bitdefender"],
        "Sophos": ["sophos"],
        "Eset Nod32": ["eset", "nod32", "Nod32", "Eset Security", "Eset", "ESET"],
        "Panda Security": ["panda", "Panda"],
    }
}

CNAE_MAP = {
    "6920": "Servicios", 
    "6201": "Tecnología", 
    "6202": "Tecnología", 
    "4120": "Industrial",
    "2222": "Industrial",
    "1721": "Industrial",
    "2849": "Industrial",
    "6022": "Servicios",
}

SURVEY_SCHEMA = {
    # --- 0. DIMENSIÓN EMPRESARIAL ---
    "company_profile_cnae": ["Industrial", "Servicios", "Comercio", "Tecnología", "Otro"],
    "number_of_employees": ["NUMERIC"],
    "average_employee_age": ["<30", "30-40", "41-50", ">50"],
    "number_of_clients": ["NUMERIC"],
    "number_of_suppliers": ["NUMERIC"],
    "annual_revenue": ["<1M", "1-5M", "5-20M", ">20M"],
    "it_outsourcing_level": ["Bajo", "Medio", "Alto"],

    # --- 1. INFRAESTRUCTURA ---
    "remote_work_acceptable_use_policy": ["Sí", "No", "En desarrollo"],
    "secure_remote_access": ["Sí", "No", "Parcial"],
    "two_factor_authentication": ["Sí", "No", "Parcial"],
    "it_infrastructure_type": ["On-premise", "Híbrida", "Cloud"],

    # --- 2. PROCESOS ---
    "key_processes_digitized_pct": ["0-25%", "26-50%", "51-75%", "76-100%"],
    "erp_in_use": ["SAP", "Microsoft", "Oracle", "Odoo", "Otro", "Ninguno"],
    "crm_in_use": ["Salesforce", "HubSpot", "Zoho", "Otro", "Ninguno"],
    "ai_for_automation_usage": ["No", "Explorando", "En piloto", "En producción"],

    # --- 3. DATOS ---
    "database_type": ["Cloud", "OnPremise"],
    "powerbi_usage": ["Power BI", "Tableau", "Excel", "Otro", "Ninguno"],

    # --- 4. PERSONAS ---
    "advanced_digital_skills_pct": ["0-25%", "26-50%", "51-75%", "76-100%"],
    "microsoft_365_usage": ["Sí", "No"],
    "collaboration_tools_usage": ["Sí", "No", "Parcial"],
    "continuous_digital_training": ["Sí", "No", "Ocasional"],
    "cybersecurity_training": ["Sí", "No", "<1 vez al año"],
    "ftfe_training": ["Sí", "No"],
    "phishing_simulations": ["Sí", "No", "Puntual"],

    # --- 5. PRESENCIA INTERNET ---
    "active_internet_presence": ["Sí", "No", "Parcial"],
    "active_social_media_management": ["Sí", "No", "Ocasional"],
    "digital_marketing_use": ["Sí", "No", "En evaluación"],
    "visitor_follower_analysis": ["Sí", "No", "Parcial"],

    # --- 6. VENTAS ONLINE ---
    "accessible_digital_sales_channels": ["Sí", "No", "En desarrollo"],
    "digital_revenue": ["<10%", "10-30%", "30-60%", ">60%"],
    "usual_customer_communication_channel": ["TEXT"],
    "preferred_customer_communication_channel": ["TEXT"],

    # --- 7. CIBERSEGURIDAD ---
    "antivirus_used": ["TEXT"],
    "employees_using_antivirus_pct": ["NUMERIC"],
    "regular_patching_and_updates": ["Sí", "No", "Parcial"],
    "network_controls_implemented": ["Sí", "No", "Parcial"],

    # --- 8, 9, 10. GESTIÓN Y LEGAL ---
    "documented_account_lifecycle_process": ["Sí", "No", "Parcial"],
    "clear_roles_and_privileges": ["Sí", "No", "Parcial"],
    "incident_response_plan": ["Sí", "No", "En elaboración"],
    "continuity_and_recovery_plans": ["Sí", "No", "En desarrollo"],
    "data_protection_compliance": ["Sí", "No", "Parcial"],
    "legal_and_compliance_training": ["Sí", "No", "Parcial"],
    
    # --- PREGUNTA FINAL ---
    "priority_assessment_area": [
        "Infraestructuras y conectividad", "Procesos y automatización", "Datos y analítica",
        "Personas y cultura digital", "Presencia en Internet y redes sociales",
        "Canales de venta online y experiencia del usuario", "Ciberseguridad",
        "Gestión de identidades y control de accesos",
        "Gestión de incidencias y continuidad de negocio",
        "Protección de datos y propiedad intelectual"
    ]
}

QUEST_MAPPING = {
    "perfil_empresa_cnae": "company_profile_cnae",
    "codigo_postal_de_la_empresa": "company_postcode",
    "numero_de_trabajadores": "number_of_employees",
    "edad_media_de_los_trabajadores": "average_employee_age",
    "numero_de_clientes": "number_of_clients",
    "numero_de_proveedores": "number_of_suppliers",
    "facturacion_anual": "annual_revenue",
    "nivel_de_externalizacion_ti": "it_outsourcing_level",
    "existe_politica_de_uso_aceptable_para_el_teletraba": "remote_work_acceptable_use_policy",
    "acceso_remoto_seguro": "secure_remote_access",
    "verificacion_de_doble_factor": "two_factor_authentication",
    "tipo_de_infraestructura_ti": "it_infrastructure_type",
    "_de_procesos_clave_digitalizados": "key_processes_digitized_pct",
    "que_erp_utilizas": "erp_in_use",
    "que_crm_utilizas": "crm_in_use",
    "uso_de_ia_para_automatizacion": "ai_for_automation_usage",
    "tipo_de_base_de_datos_utilizada": "database_type",
    "usas_powerbi_para_la_analitica_empresarias_de_tu_n": "powerbi_usage",
    "_empleados_con_competencias_digitales_avanzadas": "advanced_digital_skills_pct",
    "uso_de_la_suite_microsoft_": "microsoft_365_usage",
    "uso_de_herramientas_colaborativas": "collaboration_tools_usage",
    "formacion_continua_digital": "continuous_digital_training",
    "formacion_en_ciberseguridad": "cybersecurity_training",
    "formacion_ftfe_bonificada": "ftfe_training",
    "simulacros_o_tests_de_phishing": "phishing_simulations",
    "presencia_activa_en_internet": "active_internet_presence",
    "gestion_activa_de_redes_sociales": "active_social_media_management",
    "uso_de_marketing_digital": "digital_marketing_use",
    "analisis_de_visitantes_y_seguidores": "visitor_follower_analysis",
    "canales_de_venta_digitales_accesibles": "accessible_digital_sales_channels",
    "ingresos_digitales": "digital_revenue",
    "canal_habitual_comunicacion_cliente": "usual_customer_communication_channel",
    "canal_preferente_comunicacion_cliente": "preferred_customer_communication_channel",
    "que_antivirus_usan_los_empleados": "antivirus_used",
    "que_porcentaje_de_tus_empleados_usan_antivirus": "employees_using_antivirus_pct",
    "aplicacion_regular_de_parches_y_actualizaciones": "regular_patching_and_updates",
    "controles_de_red_implementados": "network_controls_implemented",
    "proceso_documentado_de_alta_modificacion_y_baja_de": "documented_account_lifecycle_process",
    "asignacion_clara_de_roles_y_privilegios": "clear_roles_and_privileges",
    "plan_de_respuesta_a_incidentes_aprobado": "incident_response_plan",
    "planes_de_continuidad_y_recuperacion": "continuity_and_recovery_plans",
    "cumplimiento_con_normativa_de_proteccion_de_datos": "data_protection_compliance",
    "formacion_en_cumplimiento_legal_y_normativo_digita": "legal_and_compliance_training",
    "de_los_ambitos_evaluados_cual_considera_que_es_pri": "priority_assessment_area"
}

SCORING_MAPS = {
    "binary": {
        'Sí': 100,
        'Parcial': 50, 'En desarrollo': 50, 
        'Ocasional': 50, 'En evaluación': 50, 
        'En elaboración': 50,
        'No': 0, 'Ninguno': 0
    },
    
    "percentage": {
        '0-25%': 25, 
        '26-50%': 50, 
        '51-75%': 75, 
        '76-100%': 100
    },
    
    "infrastructure": {
        'Cloud': 100, 
        'Híbrida': 70, 
        'On-premise': 30
    },
    
    "ai": {
        'En producción': 100, 
        'En piloto': 75, 
        'Explorando': 40, 
        'No': 0
    },
    
    "revenue": {
        '>60%': 100, 
        '30-60%': 75, 
        '10-30%': 50, 
        '<10%': 25
    }
}