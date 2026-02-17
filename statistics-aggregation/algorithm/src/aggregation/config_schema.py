KEYWORD_RULES = {
    "software": {
        "A3 / Wolters Kluwer": ["a3", "wolter", "kluwer", "a3erp"],
        "Sage": ["sage", "murano", "sage50", "sage200", "eurowin"],
        "Microsoft Dynamics": ["navision", "nav", "business central", "dynamics", "bc"],
        "Excel": ["excel", "sheet", "hoja", "calculo", "office", "microsoft"],
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
        "Microsoft Defender": ["defender", "windows", "microsoft", "no usan", "ninguno", "nvidia"],
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
    "694.1": "Otro",
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

SECTION_MAPPING = {
    # --- 0. DIMENSIÓN EMPRESARIAL ---
    "company_profile_cnae": "0",
    "number_of_employees": "0",
    "average_employee_age": "0",
    "number_of_clients": "0",
    "number_of_suppliers": "0",
    "annual_revenue": "0",
    "it_outsourcing_level": "0",
    
    "remote_work_acceptable_use_policy": "1",
    "secure_remote_access": "1",
    "two_factor_authentication": "1",
    "it_infrastructure_type": "1",
    
    # --- 2. Procesos y automatización ---
    "key_processes_digitized_pct": "2",
    "erp_in_use": "2",
    "crm_in_use": "2",
    "ai_for_automation_usage": "2",
    
    # --- 3. Datos y analítica ---
    "database_type": "3",
    "powerbi_usage": "3",
    
    # --- 4. Personas y cultura digital ---
    "advanced_digital_skills_pct": "4",
    "microsoft_365_usage": "4",
    "collaboration_tools_usage": "4",
    "continuous_digital_training": "4",
    "cybersecurity_training": "4",
    "ftfe_training": "4",
    "phishing_simulations": "4",
    
    # --- 5. PRESENCIA INTERNET ---
    "active_internet_presence": "5",
    "active_social_media_management": "5",
    "digital_marketing_use": "5",
    "visitor_follower_analysis": "5",
    
    # --- 6. VENTAS ONLINE ---
    "accessible_digital_sales_channels": "6",
    "digital_revenue": "6",
    "usual_customer_communication_channel": "6",
    "preferred_customer_communication_channel": "6",
    
    # --- 7. CIBERSEGURIDAD ---
    "antivirus_used": "7",
    "employees_using_antivirus_pct": "7",
    "regular_patching_and_updates": "7",
    "network_controls_implemented": "7",
    
    "documented_account_lifecycle_process": "8",
    "clear_roles_and_privileges": "8",

    "incident_response_plan": "9",
    "continuity_and_recovery_plans": "9",

    "data_protection_compliance": "10",
    "legal_and_compliance_training": "10",
    "priority_assessment_area": "10",
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

POSTAL_CODE = {
    "01": "Álava",
        "02": "Albacete", 
        "03": "Alicante",
        "04": "Almería",
        "05": "Ávila",
        "06": "Badajoz",
        "08": "Barcelona",
        "09": "Burgos",
        "10": "Cáceres",
        "11": "Cádiz",
        "12": "Castellón",
        "13": "Ciudad Real",
        "14": "Córdoba",
        "15": "A Coruña",
        "16": "Cuenca",
        "17": "Girona",
        "18": "Granada",
        "19": "Guadalajara",
        "20": "Gipuzkoa",
        "21": "Huelva",
        "22": "Huesca",
        "23": "Jaén",
        "24": "León",
        "25": "Lleida",
        "26": "La Rioja",
        "27": "Lugo",
        "28": "Madrid",
        "29": "Málaga",
        "30": "Murcia",
        "31": "Navarra",
        "32": "Ourense",
        "33": "Asturias",
        "34": "Palencia",
        "35": "Palmas (Las)",
        "36": "Pontevedra",
        "37": "Salamanca",
        "38": "Santa Cruz de Tenerife",
        "39": "Santander",
        "40": "Segovia",
        "41": "Sevilla",
        "42": "Soria",
        "43": "Tarragona",
        "44": "Teruel",
        "45": "Toledo",
        "46": "Valencia",
        "47": "Valladolid",
        "48": "Vizcaya",
        "49": "Zamora",
        "50": "Zaragoza",
        "51": "Ceuta",
        "52": "Melilla"
}

REGION_COORDINATES = {
    # --- Andalucía ---
    'Almería': {'lat': 36.8340, 'lon': -2.4637},
    'Cádiz': {'lat': 36.5271, 'lon': -6.2886},
    'Córdoba': {'lat': 37.8882, 'lon': -4.7663},
    'Granada': {'lat': 37.1773, 'lon': -3.5898},
    'Huelva': {'lat': 37.2614, 'lon': -6.9447},
    'Jaén': {'lat': 37.7796, 'lon': -3.7849},
    'Málaga': {'lat': 36.7196, 'lon': -4.4203},
    'Sevilla': {'lat': 37.3891, 'lon': -5.9845},

    # --- Aragón ---
    'Huesca': {'lat': 42.1361, 'lon': -0.4087},
    'Teruel': {'lat': 40.3456, 'lon': -1.1065},
    'Zaragoza': {'lat': 41.6488, 'lon': -0.8891},

    # --- Asturias ---
    'Asturias': {'lat': 43.3619, 'lon': -5.8494}, # Oviedo

    # --- Baleares ---
    'Palma de Mallorca': {'lat': 39.5696, 'lon': 2.6502},
    'Illes Balears': {'lat': 39.6953, 'lon': 3.0176}, # Por si usas el nombre de provincia

    # --- Canarias ---
    'Las Palmas': {'lat': 28.1235, 'lon': -15.4363},
    'Santa Cruz de Tenerife': {'lat': 28.4674, 'lon': -16.2538},

    # --- Cantabria ---
    'Cantabria': {'lat': 43.4623, 'lon': -3.8050}, # Santander

    # --- Castilla-La Mancha ---
    'Albacete': {'lat': 38.9943, 'lon': -1.8585},
    'Ciudad Real': {'lat': 38.9848, 'lon': -3.9274},
    'Cuenca': {'lat': 40.0704, 'lon': -2.1374},
    'Guadalajara': {'lat': 40.6328, 'lon': -3.1602},
    'Toledo': {'lat': 39.8564, 'lon': -4.0199},

    # --- Castilla y León ---
    'Ávila': {'lat': 40.6565, 'lon': -4.7019},
    'Burgos': {'lat': 42.3439, 'lon': -3.6969},
    'León': {'lat': 42.5987, 'lon': -5.5671},
    'Palencia': {'lat': 42.0095, 'lon': -4.5286},
    'Salamanca': {'lat': 40.9701, 'lon': -5.6635},
    'Segovia': {'lat': 40.9429, 'lon': -4.1088},
    'Soria': {'lat': 41.7666, 'lon': -2.4790},
    'Valladolid': {'lat': 41.6523, 'lon': -4.7245},
    'Zamora': {'lat': 41.5063, 'lon': -5.7446},

    # --- Catalunya ---
    'Barcelona': {'lat': 41.3851, 'lon': 2.1734},
    'Girona': {'lat': 41.9794, 'lon': 2.8214},
    'Lleida': {'lat': 41.6176, 'lon': 0.6200},
    'Tarragona': {'lat': 41.1189, 'lon': 1.2445},

    # --- Comunitat Valenciana ---
    'Alicante': {'lat': 38.3452, 'lon': -0.4810},
    'Castellón': {'lat': 39.9864, 'lon': -0.0513},
    'Valencia': {'lat': 39.4699, 'lon': -0.3763},

    # --- Extremadura ---
    'Badajoz': {'lat': 38.8797, 'lon': -6.9706},
    'Cáceres': {'lat': 39.4753, 'lon': -6.3723},

    # --- Galicia ---
    'A Coruña': {'lat': 43.3724, 'lon': -8.3898},
    'Lugo': {'lat': 43.0121, 'lon': -7.5558},
    'Ourense': {'lat': 42.3358, 'lon': -7.8639},
    'Pontevedra': {'lat': 42.4299, 'lon': -8.6446},

    # --- Madrid ---
    'Madrid': {'lat': 40.4168, 'lon': -3.7038},

    # --- Murcia ---
    'Murcia': {'lat': 37.9922, 'lon': -1.1307},

    # --- Navarra ---
    'Navarra': {'lat': 42.8125, 'lon': -1.6458}, # Pamplona

    # --- País Vasco ---
    'Bilbao': {'lat': 43.2630, 'lon': -2.9350},
    'Bizkaia': {'lat': 43.2630, 'lon': -2.9350}, # Nombre en euskera
    'Gipuzkoa': {'lat': 43.3183, 'lon': -1.9812}, # San Sebastián
    'Álava': {'lat': 42.8467, 'lon': -2.6716},   # Vitoria
    'Araba': {'lat': 42.8467, 'lon': -2.6716},   # Nombre en euskera

    # --- La Rioja ---
    'La Rioja': {'lat': 42.4627, 'lon': -2.4450},

    # --- Ciudades Autónomas ---
    'Ceuta': {'lat': 35.8894, 'lon': -5.3213},
    'Melilla': {'lat': 35.2923, 'lon': -2.9381},

    # --- Fallback ---
    'Unknown': {'lat': 40.0000, 'lon': -4.0000}
}