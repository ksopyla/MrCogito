# NER Labels for Multi-Word Token (MWT) Extraction

## Goal
Extend tokenizer vocabulary with meaningful multi-word expressions that represent atomic concepts for the Concept Encoder architecture.

## Label Taxonomy (71 Entity Types)

### 1. People & Roles (8 labels)
- **Person** - Full names: "Albert Einstein", "Marie Curie", "Abraham Lincoln"
- **Scientist** - Scientific figures: "Isaac Newton", "Charles Darwin"
- **Historical Figure** - Historical persons: "Julius Caesar", "Cleopatra"
- **Author** - Writers/researchers: "William Shakespeare", "Stephen King"
- **Job Title** - Professional roles: "Chief Executive Officer", "Software Engineer"
- **Academic Title** - Degrees: "Doctor of Philosophy", "Master of Science"
- **Military Rank** - Ranks: "General Patton", "Admiral Nelson"
- **Political Figure** - Leaders: "Prime Minister", "President Lincoln"

### 2. Geographic & Political (9 labels)
- **Location** - General places: "New York", "San Francisco", "Mount Everest"
- **Country** - Nations: "United States", "United Kingdom", "People's Republic of China"
- **City** - Urban areas: "Los Angeles", "Buenos Aires"
- **Region** - Geographic regions: "Middle East", "Sub-Saharan Africa"
- **Landmark** - Notable places: "Eiffel Tower", "Great Wall"
- **Body of Water** - Oceans/lakes: "Pacific Ocean", "Lake Baikal"
- **Political Entity** - Unions/alliances: "European Union", "United Nations"
- **Government Agency** - Agencies: "Federal Bureau of Investigation", "National Aeronautics and Space Administration"
- **Mountain Range** - Ranges: "Rocky Mountains", "Himalayas"

### 3. Organizations & Institutions (7 labels)
- **Organization** - General orgs: "World Health Organization"
- **Company** - Corporations: "Microsoft Corporation", "Google LLC"
- **University** - Educational: "Massachusetts Institute of Technology", "Stanford University"
- **Research Institution** - Labs: "Los Alamos National Laboratory"
- **Non-Profit** - NGOs: "Red Cross", "Doctors Without Borders"
- **Sports Team** - Teams: "New York Yankees", "Manchester United"
- **Conference** - Academic: "Neural Information Processing Systems", "International Conference on Machine Learning"

### 4. Mathematical & Scientific Concepts (12 labels)
- **Mathematical Concept** - Terms: "prime number", "common denominator", "Pythagorean theorem"
- **Mathematical Operation** - Operations: "least common multiple", "greatest common divisor"
- **Statistical Measure** - Stats: "standard deviation", "confidence interval", "p-value"
- **Scientific Theory** - Theories: "theory of relativity", "natural selection", "quantum mechanics"
- **Physical Law** - Laws: "Newton's laws", "thermodynamic laws", "conservation of energy"
- **Chemical Compound** - Compounds: "carbon dioxide", "sodium chloride", "deoxyribonucleic acid"
- **Biological Entity** - Bio terms: "amino acid", "gene expression", "cell membrane"
- **Astronomical Object** - Space: "Milky Way", "black hole", "neutron star"
- **Geological Era** - Time periods: "Jurassic Period", "Ice Age"
- **Disease** - Medical: "Alzheimer's disease", "type 2 diabetes"
- **Medical Procedure** - Procedures: "computed tomography", "magnetic resonance imaging"
- **Anatomical Structure** - Anatomy: "central nervous system", "cardiovascular system"

### 5. Historical & Temporal (6 labels)
- **Historical Event** - Events: "World War II", "French Revolution", "Cold War"
- **Historical Period** - Eras: "Industrial Revolution", "Renaissance", "Bronze Age"
- **Battle** - Military: "Battle of Waterloo", "Battle of Gettysburg"
- **Treaty** - Agreements: "Treaty of Versailles", "Paris Agreement"
- **Dynasty** - Ruling families: "Ming Dynasty", "Roman Empire"
- **Social Movement** - Movements: "Civil Rights Movement", "Women's Suffrage"

### 6. Technology & Computing (11 labels)
- **Programming Language** - Languages: "Python programming", "C plus plus"
- **Software Library** - Libraries: "TensorFlow library", "React framework"
- **Algorithm** - Algorithms: "quicksort algorithm", "gradient descent", "binary search"
- **Data Structure** - Structures: "hash table", "binary tree", "linked list"
- **File Format** - Formats: "Portable Document Format", "Joint Photographic Experts Group"
- **Protocol** - Protocols: "Hypertext Transfer Protocol", "Transmission Control Protocol"
- **Operating System** - OS: "Windows operating system", "Linux kernel"
- **Database System** - DBs: "relational database", "NoSQL database"
- **Machine Learning Method** - ML: "convolutional neural network", "random forest", "support vector machine"
- **Encryption Method** - Crypto: "public key cryptography", "Advanced Encryption Standard"
- **Software Architecture** - Patterns: "model view controller", "microservices architecture"

### 7. Legal & Regulatory (4 labels)
- **Legal Term** - Terminology: "due process", "habeas corpus", "eminent domain"
- **Law** - Legislation: "Sherman Antitrust Act", "General Data Protection Regulation"
- **Legal Case** - Cases: "Brown v Board of Education"
- **Patent** - Patents: "U.S. Patent", "European Patent"

### 8. Arts & Culture (5 labels)
- **Artwork** - Art: "Mona Lisa", "Starry Night"
- **Book Title** - Books: "War and Peace", "Origin of Species"
- **Musical Work** - Music: "Symphony No. 9", "Bohemian Rhapsody"
- **Film Title** - Movies: "Gone with the Wind"
- **Cultural Movement** - Movements: "Romantic movement", "Impressionism"

### 9. Economics & Finance (4 labels)
- **Economic Concept** - Terms: "supply and demand", "gross domestic product"
- **Financial Instrument** - Instruments: "exchange-traded fund", "mortgage-backed security"
- **Economic Event** - Events: "Great Depression", "Financial Crisis"
- **Currency** - Money: "US Dollar", "British Pound"

### 10. Miscellaneous Concepts (5 labels)
- **Sport** - Sports: "association football", "American football"
- **Measurement Unit** - Units: "meters per second", "kilowatt hour"
- **Award** - Prizes: "Nobel Prize", "Pulitzer Prize", "Academy Award"
- **Holiday** - Occasions: "Independence Day", "Christmas Day"
- **Food** - Cuisine: "ice cream", "French fries"

---

## Priority Tiers for Implementation

### Tier 1 (High Value - Implement First)
These capture the most semantically dense concepts:
```
Person, Location, Organization, Historical Event, Scientific Theory,
Mathematical Concept, Algorithm, Programming Language, Disease,
Chemical Compound
```

### Tier 2 (Medium Value)
Domain-specific but important:
```
Company, University, Country, City, Machine Learning Method,
Physical Law, Statistical Measure, Data Structure, Book Title,
Economic Concept
```

### Tier 3 (Lower Priority)
Nice-to-have for completeness:
```
Artwork, Musical Work, Sport, Food, Holiday, Cultural Movement
```

---

## Estimated Coverage

Using GLiNER with these labels on Minipile:
- **Expected MWTs**: 500-2000 unique proper noun phrases
- **Quality**: 95%+ (GLiNER is trained to avoid false positives)
- **Speed**: ~10-20 minutes on 100K samples (batch processing)

## Implementation Strategy

1. **Start with Tier 1** (10 labels) - Run GLiNER on 50K samples
2. **Analyze results** - See which labels produce the most valuable MWTs
3. **Expand to Tier 2** - Add 15 more labels based on findings
4. **Combine with Statistical** - Merge GLiNER proper nouns + Statistical collocations

---

