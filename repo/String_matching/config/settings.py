"""
Configuration settings for string matching.
Contains all constants, mappings, and thresholds.
"""

# =============================================================================
# Date Tolerance
# =============================================================================
DATE_TOLERANCE_DAYS = 0

# =============================================================================
# Similarity Thresholds for X-TPA Linkage
# =============================================================================
X_TPA_NAME_THRESHOLD = 0.65
X_TPA_DESC_THRESHOLD = 0.30
X_TPA_STATE_THRESHOLD = 0.8

# =============================================================================
# Similarity Thresholds for TPA Clustering
# =============================================================================
TPA_NAME_THRESHOLD = 0.40
TPA_DESC_THRESHOLD = 0.30
TPA_STATE_THRESHOLD = 0.8

# =============================================================================
# Coverage Mapping
# =============================================================================
COVERAGE_MAP = {
    "Workers Compensation": "WC",
    "Workers Compensation Excess": "XW",
    "Auto Liability": "AU",
    "General Liability": "GL",
    "Auto Physical Damage": "AU",
}

# =============================================================================
# US State Abbreviations
# =============================================================================
STATE_ABBREV = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", 
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", 
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", 
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", 
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY", "DC"
}

# =============================================================================
# US City to State Mapping (abbreviated - extend as needed)
# =============================================================================
US_CITY_ST_DICT = {
    'NEW YORK': 'NY',
    'LOS ANGELES': 'CA',
    'CHICAGO': 'IL',
    'HOUSTON': 'TX',
    'PHOENIX': 'AZ',
    'PHILADELPHIA': 'PA',
    'SAN ANTONIO': 'TX',
    'SAN DIEGO': 'CA',
    'DALLAS': 'TX',
    'SAN JOSE': 'CA',
    'AUSTIN': 'TX',
    'JACKSONVILLE': 'FL',
    'FORT WORTH': 'TX',
    'COLUMBUS': 'OH',
    'CHARLOTTE': 'NC',
    'SAN FRANCISCO': 'CA',
    'INDIANAPOLIS': 'IN',
    'SEATTLE': 'WA',
    'DENVER': 'CO',
    'WASHINGTON': 'DC',
    'BOSTON': 'MA',
    'NASHVILLE': 'TN',
    'DETROIT': 'MI',
    'PORTLAND': 'OR',
    'MEMPHIS': 'TN',
    'OKLAHOMA CITY': 'OK',
    'LAS VEGAS': 'NV',
    'LOUISVILLE': 'KY',
    'BALTIMORE': 'MD',
    'MILWAUKEE': 'WI',
    'ALBUQUERQUE': 'NM',
    'TUCSON': 'AZ',
    'FRESNO': 'CA',
    'SACRAMENTO': 'CA',
    'ATLANTA': 'GA',
    'KANSAS CITY': 'MO',
    'LONG BEACH': 'CA',
    'MESA': 'AZ',
    'RALEIGH': 'NC',
    'OMAHA': 'NE',
    'MIAMI': 'FL',
    'OAKLAND': 'CA',
    'MINNEAPOLIS': 'MN',
    'TULSA': 'OK',
    'CLEVELAND': 'OH',
    'WICHITA': 'KS',
    'ARLINGTON': 'TX',
    'NEW ORLEANS': 'LA',
    'BAKERSFIELD': 'CA',
    'TAMPA': 'FL',
    'HONOLULU': 'HI',
    'ANAHEIM': 'CA',
    'AURORA': 'CO',
    'SANTA ANA': 'CA',
    'ST LOUIS': 'MO',
    'RIVERSIDE': 'CA',
    'CORPUS CHRISTI': 'TX',
    'PITTSBURGH': 'PA',
    'LEXINGTON': 'KY',
    'ANCHORAGE': 'AK',
    'STOCKTON': 'CA',
    'CINCINNATI': 'OH',
    'SAINT PAUL': 'MN',
    'TOLEDO': 'OH',
    'NEWARK': 'NJ',
    'GREENSBORO': 'NC',
    'PLANO': 'TX',
    'HENDERSON': 'NV',
    'LINCOLN': 'NE',
    'BUFFALO': 'NY',
    'FORT WAYNE': 'IN',
    'JERSEY CITY': 'NJ',
    'CHULA VISTA': 'CA',
    'ORLANDO': 'FL',
    'ST PETERSBURG': 'FL',
    'NORFOLK': 'VA',
    'CHANDLER': 'AZ',
    'LAREDO': 'TX',
    'MADISON': 'WI',
    'DURHAM': 'NC',
    'LUBBOCK': 'TX',
    'WINSTON SALEM': 'NC',
    'GARLAND': 'TX',
    'GLENDALE': 'AZ',
    'HIALEAH': 'FL',
    'RENO': 'NV',
    'BATON ROUGE': 'LA',
    'IRVINE': 'CA',
    'CHESAPEAKE': 'VA',
    'IRVING': 'TX',
    'SCOTTSDALE': 'AZ',
    'NORTH LAS VEGAS': 'NV',
    'FREMONT': 'CA',
    'GILBERT': 'AZ',
    'SAN BERNARDINO': 'CA',
    'BOISE': 'ID',
    'BIRMINGHAM': 'AL',
}

# =============================================================================
# USA Names Nicknames Mapping
# =============================================================================
USA_NAMES_NICKNAMES = {
    'William': ['Bill', 'Will', 'Liam', 'Billy', 'Willy'],
    'Robert': ['Rob', 'Bob', 'Bobby', 'Robbie', 'Bert'],
    'Richard': ['Rick', 'Dick', 'Rich', 'Ricky', 'Richie'],
    'James': ['Jim', 'Jimmy', 'Jamie', 'Jimbo'],
    'John': ['Jack', 'Johnny', 'Jon'],
    'Michael': ['Mike', 'Mikey', 'Mick', 'Mickey'],
    'David': ['Dave', 'Davey', 'Davie'],
    'Joseph': ['Joe', 'Joey', 'Jo'],
    'Thomas': ['Tom', 'Tommy', 'Thom'],
    'Charles': ['Charlie', 'Chuck', 'Chas', 'Chaz'],
    'Christopher': ['Chris', 'Topher', 'Kit'],
    'Daniel': ['Dan', 'Danny', 'Dani'],
    'Matthew': ['Matt', 'Matty'],
    'Anthony': ['Tony', 'Ant', 'Anton'],
    'Donald': ['Don', 'Donnie', 'Donny'],
    'Steven': ['Steve', 'Stevie'],
    'Stephen': ['Steve', 'Stevie'],
    'Paul': ['Paulie'],
    'Andrew': ['Andy', 'Drew', 'Dre'],
    'Joshua': ['Josh'],
    'Kenneth': ['Ken', 'Kenny'],
    'Kevin': ['Kev'],
    'Brian': ['Bri'],
    'George': ['Geo', 'Georgie'],
    'Timothy': ['Tim', 'Timmy'],
    'Ronald': ['Ron', 'Ronnie', 'Ronny'],
    'Edward': ['Ed', 'Eddie', 'Ted', 'Teddy', 'Ned'],
    'Jason': ['Jay', 'Jase'],
    'Jeffrey': ['Jeff', 'Jeffy'],
    'Ryan': ['Ry'],
    'Jacob': ['Jake', 'Jakey'],
    'Gary': ['Gar'],
    'Nicholas': ['Nick', 'Nicky', 'Nico'],
    'Eric': ['Rick', 'Ricky'],
    'Jonathan': ['Jon', 'Johnny', 'Nate'],
    'Patrick': ['Pat', 'Patty', 'Paddy'],
    'Frank': ['Frankie', 'Franky'],
    'Scott': ['Scotty'],
    'Benjamin': ['Ben', 'Benny', 'Benji'],
    'Samuel': ['Sam', 'Sammy'],
    'Gregory': ['Greg', 'Gregg'],
    'Alexander': ['Alex', 'Xander', 'Lex', 'Alec'],
    'Raymond': ['Ray'],
    'Peter': ['Pete', 'Petey'],
    'Henry': ['Hank', 'Harry', 'Hal'],
    'Douglas': ['Doug', 'Dougie'],
    'Aaron': ['Ari', 'Ron'],
    'Katherine': ['Kate', 'Katie', 'Kathy', 'Kay', 'Kat', 'Kitty'],
    'Elizabeth': ['Liz', 'Lizzy', 'Beth', 'Betsy', 'Betty', 'Eliza', 'Libby'],
    'Jennifer': ['Jen', 'Jenny', 'Jenn'],
    'Margaret': ['Maggie', 'Meg', 'Peggy', 'Marge', 'Margie', 'Margo'],
    'Susan': ['Sue', 'Susie', 'Suzy'],
    'Jessica': ['Jess', 'Jessie'],
    'Sarah': ['Sally', 'Sadie'],
    'Karen': ['Kari', 'Karrie'],
    'Nancy': ['Nan', 'Nannie'],
    'Betty': ['Bettie', 'Betts'],
    'Dorothy': ['Dot', 'Dottie', 'Dolly'],
    'Lisa': ['Lis'],
    'Sandra': ['Sandy', 'Sandi'],
    'Ashley': ['Ash'],
    'Kimberly': ['Kim', 'Kimmy', 'Kimmie'],
    'Donna': ['Donnie'],
    'Emily': ['Em', 'Emmy', 'Emmie'],
    'Michelle': ['Micki', 'Mish', 'Shell', 'Shelly'],
    'Carol': ['Carole', 'Carrie'],
    'Amanda': ['Mandy', 'Manda'],
    'Melissa': ['Mel', 'Missy', 'Lisa'],
    'Deborah': ['Deb', 'Debbie', 'Debby'],
    'Stephanie': ['Steph', 'Stephie', 'Stevie'],
    'Rebecca': ['Becca', 'Becky', 'Reba'],
    'Sharon': ['Shari', 'Sherry'],
    'Laura': ['Laurie', 'Lori'],
    'Cynthia': ['Cindy', 'Cindi', 'Cyndi'],
    'Kathleen': ['Kate', 'Katie', 'Kathy', 'Kay'],
    'Amy': ['Aimee'],
    'Angela': ['Angie', 'Angel'],
    'Shirley': ['Shirl'],
    'Anna': ['Annie', 'Ann'],
    'Brenda': ['Bren'],
    'Pamela': ['Pam', 'Pammy'],
    'Emma': ['Em', 'Emmy'],
    'Nicole': ['Nikki', 'Nicky', 'Nicki', 'Cole'],
    'Helen': ['Ellie', 'Nellie', 'Nell'],
    'Samantha': ['Sam', 'Sammy', 'Sammie'],
    'Victoria': ['Vicky', 'Vicki', 'Tori', 'Vic'],
    'Christine': ['Chris', 'Christy', 'Tina'],
    'Christina': ['Chris', 'Christy', 'Tina'],
    'Rachel': ['Rae', 'Rach'],
    'Janet': ['Jan'],
    'Catherine': ['Cathy', 'Kate', 'Katie', 'Cat'],
    'Maria': ['Mary', 'Mia'],
    'Heather': ['Heath'],
    'Diana': ['Di', 'Diane'],
    'Judith': ['Judy', 'Judi'],
    'Julie': ['Jules'],
    'Olivia': ['Liv', 'Livvy', 'Ollie'],
    'Joyce': ['Joy'],
    'Virginia': ['Ginny', 'Ginger', 'Virgie'],
    'Jacqueline': ['Jackie', 'Jacqui'],
    'Theresa': ['Terry', 'Terri', 'Tess', 'Tessie'],
    'Grace': ['Gracie'],
    'Teresa': ['Terry', 'Terri'],
    'Ann': ['Annie', 'Anna'],
    'Sara': ['Sally'],
    'Gloria': ['Glory'],
    'Janice': ['Jan'],
    'Jean': ['Jeanie', 'Jeannie'],
    'Abigail': ['Abby', 'Abbie', 'Gail'],
    'Alice': ['Ali', 'Ally', 'Allie'],
    'Judy': ['Judi'],
    'Sophia': ['Sophie'],
    'Madison': ['Maddie', 'Madi'],
    'Hannah': ['Han'],
    'Natalie': ['Nat', 'Nattie'],
    'Evelyn': ['Eve', 'Evie', 'Lyn'],
    'Megan': ['Meg'],
    'Lauren': ['Laurie'],
    'Andrea': ['Andi', 'Andie', 'Drea'],
    'Denise': ['Deni', 'Denny'],
    'Marilyn': ['Mary', 'Lynn'],
    'Amber': ['Amby'],
    'Danielle': ['Dani', 'Danni', 'Elle'],
    'Brittany': ['Brit', 'Britt', 'Britty'],
    'Carolyn': ['Carol', 'Lyn'],
    'Janet': ['Jan', 'Jannie'],
    'Frances': ['Fran', 'Franny', 'Francie'],
    'Eleanor': ['Ellie', 'Nell', 'Nelly', 'Nora'],
    'Cheryl': ['Cher', 'Cheri'],
    'Mildred': ['Millie', 'Milly'],
    'Lillian': ['Lilly', 'Lily', 'Lil'],
    'Phyllis': ['Phil'],
    'Norma': ['Norm'],
    'Paula': ['Polly'],
    'Irene': ['Rene', 'Reenie'],
    'Josephine': ['Jo', 'Josie', 'Josey'],
}

# Build reverse lookup: nickname -> formal names
NICKNAME_TO_FORMAL = {}
for formal, nicknames in USA_NAMES_NICKNAMES.items():
    formal_upper = formal.upper()
    NICKNAME_TO_FORMAL[formal_upper] = {formal_upper}
    for nick in nicknames:
        nick_upper = nick.upper()
        if nick_upper not in NICKNAME_TO_FORMAL:
            NICKNAME_TO_FORMAL[nick_upper] = set()
        NICKNAME_TO_FORMAL[nick_upper].add(formal_upper)
        NICKNAME_TO_FORMAL[formal_upper].add(nick_upper)

# =============================================================================
# File Paths (Configure for your environment)
# =============================================================================
# Databricks paths - modify for local testing if needed
TPA_CLAIMS_PATH = "/Workspace/Users/X@X.com/Claim_Linkage/data_imports/TK/TK Elevator Loss Runs 7.17.25_cleaned.xlsm"
X_CLAIMS_PATH = "/Workspace/Users/X@X.com/Claim_Linkage/data_imports/TK/TK Elevator - GRA Loss Runs Claim Detail File.xlsx"

# Output paths
OUTPUT_DIR = "/Workspace/Users/X@X.com/Claim_Linkage/outputs/TK/"

# =============================================================================
# Retry/Backoff Settings (kept for potential future API usage)
# =============================================================================
MAX_RETRIES = 5
INITIAL_BACKOFF = 1.0
