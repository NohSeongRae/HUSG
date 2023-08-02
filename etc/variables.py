government_dict = {
    'civic': 'government_office'
}

shop_dict = {
    'convenience': 'supermarket',
    'supermarket': 'supermarket',
    'herbalist': 'alternative',
    'nutrition_supplements': 'alternative',
}

amenity_dict = {
    'marketplace': 'supermarket',
    'restaurant': 'restaurant',
    'fast_food': 'restaurant',
    'cafe': 'restaurant',
    'bar': 'restaurant',
    'pub': 'restaurant',
    'kindergarten': 'kindergarten',
    'school': 'school',
    'college': 'college',
    'university': 'university',
    'police': 'police_station',
    'fire_station': 'fire_station',
    'bank': 'bank',
    'bureau_de_change': 'bureau_de_change',
    'court_house': 'government_office',
    'townhall': 'government_office',
    'embassy': 'embassy',
    'post_office': 'post_office',
    'doctors': 'clinic',
    'dentist': 'clinic',
    'clinic': 'clinic',
    'hospital': 'hospital',
    'pharmacy': 'pharmacy',
    'grave_yard': 'cemetery',
    'place_of_worship': 'place_of_worship',
    'community_centre': 'community_centre',
    'library': 'library'
}

office_dict = {
    'government': 'government_office',
}

tourism_dict = {
    'hotel': 'accommodation',
    'chalet': 'accommodation',
    'guest_house': 'accommodation',
    'hostel': 'accommodation',
    'motel': 'accommodation',
}

healthcare_dict = {
    'alternative': 'alternative'
}

leisure_dict = {
    'stadium': 'stadium',
    'swimming_pool': 'swimming_pool',
    'pitch': 'pitch',
    'sport_centre': 'sport_centre'
}


commercial_color = [0.9294117647058824, 0.8313725490196079, 0.5607843137254902, 1]
education_color = [0.8901960784313725, 0.8117647058823529, 0.6549019607843137, 1]
emergency_color = [0.9607843137254902, 0.8784313725490196, 0.8784313725490196, 1]
financial_color = [0.8980392156862745, 0.9019607843137255, 0.9215686274509803, 1]
government_color = [1.0, 0.8509803921568627, 0.4, 1]
healthcare_color = [0.9607843137254902, 0.8784313725490196, 0.8784313725490196, 1]
public_color = [0.9411764705882353, 0.9019607843137255, 0.8196078431372549, 1]
sport_color = [0.7803921568627451, 0.7803921568627451, 0.7058823529411765, 1]
residence_color = [0.803921568627451, 0.7647058823529411, 0.7411764705882353, 1]

commercial = ["shop", "supermarket", "restaurant", "tourism", "accommodation"]
education = ["kindergarten", "school", "college", "university"]
emergency = ["police_station", "ambulance_station", "fire_station"]
financial = ["bank", "bureau_de_change"]
government = ["government_office", "embassy", "military", "post_office"]
healthcare = ["doctor", "dentist", "clinic", "hospital", "pharmacy", "alternative"]
public = ["place_of_worship", "community_centre", "library", "historic", "toilet"]
sport = ["stadium", "swimming_pool", "pitch", "sport_centre"]
building = ['residence']


category_color = {}

category_list = [commercial, education, emergency, financial, government, healthcare, public, sport, building]
color_list = [commercial_color, education_color, emergency_color, financial_color, government_color, healthcare_color, public_color, sport_color, residence_color]

for i in range(len(category_list)):
    for j in range(len(category_list[i])):
        category_color[category_list[i][j]] = color_list[i]
