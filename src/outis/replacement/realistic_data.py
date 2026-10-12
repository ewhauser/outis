"""Word lists for the opt-in realistic replacement style.

Every list here was written for Outis and is distributed under the project's
licence. The town, street and institution names are built from common English
words and suffixes; they were not copied from a gazetteer, so some combinations
will match real places by coincidence. State postal codes, state names and ZIP
prefix ranges are public USPS facts, transcribed by hand; a ZIP range covers the
prefixes allocated to the state, not every prefix that is currently in use.
"""

from __future__ import annotations

FIRST_NAMES = (
    "Aaron", "Abigail", "Adam", "Alan", "Alice", "Allison", "Amanda", "Amy",
    "Andrea", "Andrew", "Angela", "Anna", "Anthony", "Ashley", "Barbara",
    "Benjamin", "Beth", "Brandon", "Brenda", "Brian", "Caleb", "Carl", "Carol",
    "Caroline", "Catherine", "Charles", "Chris", "Christina", "Claire",
    "Colin", "Connor", "Craig", "Daniel", "David", "Deborah", "Dennis", "Diana",
    "Donald", "Donna", "Dylan", "Edward", "Elaine", "Elena", "Elizabeth",
    "Emily", "Emma", "Eric", "Ethan", "Evan", "Fiona", "Frank", "Gary",
    "George", "Grace", "Gregory", "Hannah", "Heather", "Helen", "Henry", "Ian",
    "Isaac", "Jacob", "James", "Janet", "Jason", "Jeffrey", "Jennifer",
    "Jessica", "Joan", "John", "Jonathan", "Joseph", "Joshua", "Joyce", "Julia",
    "Justin", "Karen", "Katherine", "Keith", "Kelly", "Kevin", "Kyle", "Laura",
    "Lauren", "Leah", "Linda", "Lisa", "Lucas", "Lucy", "Margaret", "Maria",
    "Mark", "Martha", "Matthew", "Megan", "Melissa", "Michael", "Michelle",
    "Nancy", "Natalie", "Nathan", "Nicholas", "Nicole", "Noah", "Olivia",
    "Owen", "Patricia", "Patrick", "Paul", "Peter", "Rachel", "Raymond",
    "Rebecca", "Richard", "Robert", "Roger", "Ruth", "Ryan", "Samuel", "Sandra",
    "Sara", "Scott", "Sean", "Sharon", "Simon", "Sophia", "Stephen", "Steven",
    "Susan", "Thomas", "Timothy", "Victoria", "Vincent", "Walter", "William",
    "Zachary",
)  # fmt: skip

# Place-like stems and endings; joined they form town names and surnames such
# as "Ashford" or "Millbrook".
PLACE_STEMS = (
    "Alder", "Amber", "Arbor", "Ash", "Bay", "Beech", "Birch", "Black",
    "Bright", "Brook", "Cedar", "Clear", "Cliff", "Copper", "Crane", "Deer",
    "Dove", "East", "Elm", "Fair", "Fern", "Fox", "Glen", "Gold", "Green",
    "Hawk", "Hazel", "High", "Holly", "Ivy", "Lake", "Lark", "Laurel",
    "Linden", "Maple", "Meadow", "Mill", "Moss", "North", "Oak", "Orchard",
    "Pine", "Pond", "Quail", "Raven", "Red", "Ridge", "River", "Rock", "Rose",
    "Sand", "Silver", "South", "Spring", "Stone", "Sun", "Thorn", "West",
    "White", "Willow", "Wolf", "Wren", "Apple", "Aspen", "Barley", "Bell",
    "Bramble", "Briar", "Brush", "Chestnut", "Clover", "Cold", "Cotton",
    "Cross", "Dry", "Elder", "Falcon", "Field", "Flint", "Frost", "Hart",
    "Heather", "Hollow", "Iron", "Juniper", "Kings", "Long", "Marsh",
    "Middle", "Mist", "Rush", "Salt", "Shady", "Sparrow", "Still", "Summer",
    "Swan", "Timber",
)  # fmt: skip
PLACE_ENDINGS = (
    "bridge", "brook", "bury", "by", "crest", "dale", "field", "ford", "gate",
    "haven", "hill", "hurst", "land", "ley", "mere", "mont", "mouth", "port",
    "ridge", "side", "stead", "ton", "vale", "view", "ville", "wick", "wood",
    "worth", "bourne", "combe", "cote", "croft", "glen", "ham",
    "holm", "more", "shaw", "stone", "well", "wold", "moor",
)  # fmt: skip
# Second words of town names such as "Cedar Falls" or "Ashford Landing".
PLACE_FEATURES = (
    "Falls", "Grove", "Springs", "Hollow", "Heights", "Junction", "Point",
    "Harbor", "Crossing", "Valley", "Park", "Corners", "Landing", "Mills",
    "Creek", "Bluff", "Meadows", "Station", "Hills", "Ridge",
)  # fmt: skip
# Leading words such as "New Ashford" or "West Elmwood".
PLACE_MODIFIERS = (
    "New", "Port", "Mount", "Lake", "Glen", "North", "South", "East", "West",
    "Upper", "Lower", "Old",
)  # fmt: skip

STREET_NAMES = (
    "Oak", "Maple", "Cedar", "Elm", "Pine", "Walnut", "Chestnut", "Spruce",
    "Willow", "Hickory", "Birch", "Laurel", "Magnolia", "Juniper", "Aspen",
    "Poplar", "Sycamore", "Dogwood", "Holly", "Hawthorn", "Park", "Lake",
    "Hill", "River", "Meadow", "Spring", "Forest", "Ridge", "Valley", "Sunset",
    "Highland", "Church", "Mill", "School", "Market", "Bridge", "Center",
    "Union", "Liberty", "Orchard", "Prospect", "Garden", "Grove", "Harbor",
    "Lakeview", "Hillside", "Woodland", "Fairview", "Pleasant", "Summit",
)  # fmt: skip

# (full, abbreviation) street suffixes; the original's form is kept.
STREET_SUFFIXES = (
    ("Street", "St"),
    ("Avenue", "Ave"),
    ("Road", "Rd"),
    ("Lane", "Ln"),
    ("Drive", "Dr"),
    ("Court", "Ct"),
    ("Place", "Pl"),
    ("Way", "Way"),
    ("Boulevard", "Blvd"),
    ("Terrace", "Ter"),
    ("Circle", "Cir"),
    ("Parkway", "Pkwy"),
    ("Trail", "Trl"),
)
# Further spellings recognized in the original, but never emitted.
STREET_SUFFIX_ALIASES = frozenset(
    {"av", "hwy", "highway", "sq", "square", "aly", "alley", "row", "pike", "loop"}
)
DIRECTIONALS = frozenset(
    {
        "n", "s", "e", "w", "ne", "nw", "se", "sw", "north", "south", "east",
        "west", "northeast", "northwest", "southeast", "southwest",
    }
)  # fmt: skip
UNIT_DESIGNATORS = frozenset(
    {
        "apt", "apartment", "unit", "suite", "ste", "room", "rm", "fl", "floor",
        "bldg", "building", "lot", "space", "spc", "trlr", "dept",
    }
)  # fmt: skip
# Trailing locality words kept on a replaced town: generic administrative terms.
LOCALITY_DESIGNATORS = frozenset({"county", "township", "parish", "borough"})

FACILITY_MODIFIERS = ("", "Regional", "Community", "Memorial", "Valley")

URL_WORDS = (
    "about", "account", "appointments", "archive", "care", "contact", "docs",
    "events", "files", "forms", "health", "help", "home", "info", "library",
    "news", "notes", "portal", "profile", "records", "reports", "resources",
    "results", "search", "services", "team", "updates", "view", "visit",
    "welcome",
)  # fmt: skip
URL_EXTENSIONS = frozenset(
    {"html", "htm", "php", "asp", "aspx", "jsp", "pdf", "jpg", "jpeg", "png", "txt"}
)
# Generic host and mailbox subdomains that are safe to keep.
GENERIC_SUBDOMAINS = frozenset({"www", "mail", "email", "my", "portal", "secure"})

# Postal code -> (name, ZIP prefix ranges allocated to it).
US_STATES = {
    "AL": ("Alabama", ((350, 369),)),
    "AK": ("Alaska", ((995, 999),)),
    "AZ": ("Arizona", ((850, 865),)),
    "AR": ("Arkansas", ((716, 729),)),
    "CA": ("California", ((900, 961),)),
    "CO": ("Colorado", ((800, 816),)),
    "CT": ("Connecticut", ((60, 69),)),
    "DE": ("Delaware", ((197, 199),)),
    "DC": ("District of Columbia", ((200, 200), (202, 205))),
    "FL": ("Florida", ((320, 349),)),
    "GA": ("Georgia", ((300, 319), (398, 399))),
    "HI": ("Hawaii", ((967, 968),)),
    "ID": ("Idaho", ((832, 838),)),
    "IL": ("Illinois", ((600, 629),)),
    "IN": ("Indiana", ((460, 479),)),
    "IA": ("Iowa", ((500, 528),)),
    "KS": ("Kansas", ((660, 679),)),
    "KY": ("Kentucky", ((400, 427),)),
    "LA": ("Louisiana", ((700, 714),)),
    "ME": ("Maine", ((39, 49),)),
    "MD": ("Maryland", ((206, 219),)),
    "MA": ("Massachusetts", ((10, 27),)),
    "MI": ("Michigan", ((480, 499),)),
    "MN": ("Minnesota", ((550, 567),)),
    "MS": ("Mississippi", ((386, 397),)),
    "MO": ("Missouri", ((630, 658),)),
    "MT": ("Montana", ((590, 599),)),
    "NE": ("Nebraska", ((680, 693),)),
    "NV": ("Nevada", ((889, 898),)),
    "NH": ("New Hampshire", ((30, 38),)),
    "NJ": ("New Jersey", ((70, 89),)),
    "NM": ("New Mexico", ((870, 884),)),
    "NY": ("New York", ((100, 149),)),
    "NC": ("North Carolina", ((270, 289),)),
    "ND": ("North Dakota", ((580, 588),)),
    "OH": ("Ohio", ((430, 459),)),
    "OK": ("Oklahoma", ((730, 749),)),
    "OR": ("Oregon", ((970, 979),)),
    "PA": ("Pennsylvania", ((150, 196),)),
    "PR": ("Puerto Rico", ((6, 7), (9, 9))),
    "RI": ("Rhode Island", ((28, 29),)),
    "SC": ("South Carolina", ((290, 299),)),
    "SD": ("South Dakota", ((570, 577),)),
    "TN": ("Tennessee", ((370, 385),)),
    "TX": ("Texas", ((750, 799),)),
    "UT": ("Utah", ((840, 847),)),
    "VT": ("Vermont", ((50, 59),)),
    "VA": ("Virginia", ((220, 246),)),
    "WA": ("Washington", ((980, 994),)),
    "WV": ("West Virginia", ((247, 268),)),
    "WI": ("Wisconsin", ((530, 549),)),
    "WY": ("Wyoming", ((820, 831),)),
}
# ZIP prefixes 001-004 have never been allocated; used when no state is emitted.
UNALLOCATED_ZIP_PREFIXES = ((1, 4),)

COUNTRIES = (
    "Argentina", "Australia", "Austria", "Belgium", "Brazil", "Canada",
    "Chile", "China", "Colombia", "Denmark", "Egypt", "England", "Finland",
    "France", "Germany", "Greece", "India", "Indonesia", "Ireland", "Israel",
    "Italy", "Japan", "Kenya", "Mexico", "Morocco", "Netherlands",
    "New Zealand", "Nigeria", "Norway", "Pakistan", "Peru", "Philippines",
    "Poland", "Portugal", "Scotland", "South Africa", "South Korea", "Spain",
    "Sweden", "Switzerland", "Thailand", "Turkey", "Ukraine",
    "United Kingdom", "United States", "Vietnam", "Wales",
)  # fmt: skip
# Abbreviations recognized as a country only in this exact spelling.
COUNTRY_ABBREVIATIONS = frozenset(
    {"USA", "U.S.A.", "U.S.A", "US", "U.S.", "UK", "U.K.", "UAE"}
)

# NANP area codes for fictional numbers. Every NANP area code reserves
# 555-0100 through 555-0199 for fiction, so the area code itself only shapes the
# number. Excluded: easily recognizable codes (second digit equals third, e.g.
# 211 or 800), N9X expansion codes, toll-free 8YY codes, 456, 500-series
# personal-communication codes, 600, 700, 710, 37X and 96X.
_EXCLUDED_AREA_CODES = frozenset(
    {"456", "500", "521", "522", "523", "524", "525", "526", "527", "528",
     "529", "532", "533", "535", "538", "542", "543", "544", "545", "546",
     "547", "549", "550", "552", "553", "554", "556", "558", "566", "569",
     "577", "578", "588", "589", "600", "700", "710", "800", "833", "844",
     "855", "866", "877", "888", "900"}
)  # fmt: skip
AREA_CODES = tuple(
    code
    for code in (
        f"{n}{x}{y}" for n in range(2, 10) for x in range(9) for y in range(10)
    )
    if code[1] != code[2]
    and code not in _EXCLUDED_AREA_CODES
    and not code.startswith(("37", "96"))
)

# Never-issued SSN areas. 900-999 are never SSNs, but the IRS issues ITINs in
# 9XX with groups 50-65, 70-88, 90-92 and 94-99, so those groups are avoided.
SSN_NEVER_ISSUED_AREAS = ("000", "666") + tuple(str(area) for area in range(900, 1000))
SSN_NON_ITIN_GROUPS = tuple(range(50)) + (66, 67, 68, 69, 89, 93)
