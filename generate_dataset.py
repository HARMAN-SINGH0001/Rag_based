import os
import json
from settings import BASE_DIR

def generate_hotel_dataset():
    documents = [
        # === CATEGORY 1: Hotel Descriptions (9 docs) ===
        {
            "id": "DOC-01",
            "hotel": "Grand Plaza Hotel",
            "category": "Hotel Descriptions",
            "title": "Grand Plaza Hotel - Overview & Star Rating",
            "content": "The Grand Plaza Hotel is a premium 5-star luxury establishment located in the heart of the metropolitan downtown district. Representing the pinnacle of urban sophistication, the hotel features modern glass architecture, a magnificent 3-story marble lobby, and an elegant botanical indoor garden. Designed for both high-profile business executives and discerning leisure travelers, the Grand Plaza brand stands for uncompromising quality, refined aesthetics, and personalized service. The ambiance is formal yet warm, offering guests a peaceful sanctuary from the bustling city outside."
        },
        {
            "id": "DOC-02",
            "hotel": "Seaside Haven Resort",
            "category": "Hotel Descriptions",
            "title": "Seaside Haven Resort - Ambiance & Positioning",
            "content": "Seaside Haven Resort is a boutique 4-star seaside property positioned as a relaxing coastal getaway. The resort features low-rise Mediterranean-style villa architecture with whitewashed walls, terracotta tile roofs, and beautiful cascading bougainvillea. With a laid-back, luxurious, and family-friendly ambiance, Seaside Haven focus on wellness, nature, and recreation. It is ideal for vacationers seeking a tranquil retreat where they can listen to the waves and enjoy spectacular sunset views."
        },
        {
            "id": "DOC-03",
            "hotel": "Hotel X",
            "category": "Hotel Descriptions",
            "title": "Hotel X - Modern Business Positioning",
            "content": "Hotel X is a contemporary 4-star business-first hotel tailored for digital nomads, corporate professionals, and tech travelers. Boasting sleek, minimalist architecture and integrated smart-room technologies, Hotel X offers a productive and efficient environment. The brand positioning centers on tech-forward convenience, high-speed connectivity, and modern comfort. The atmosphere is energetic and minimalist, with shared coworking spaces in the lobby, automated check-in kiosks, and quiet meeting corners."
        },
        {
            "id": "DOC-04",
            "hotel": "Alpine Lodge",
            "category": "Hotel Descriptions",
            "title": "Alpine Lodge - Star Rating & Mountain Vibe",
            "content": "Alpine Lodge is a charming 3-star rustic mountain lodge set against the picturesque backdrop of snow-capped peaks. The property is constructed entirely of local timber and stone, evoking a warm, traditional cabin feel. Catering to outdoor enthusiasts, skiers, and nature lovers, Alpine Lodge features stone fireplaces, exposed wood beams, and cozy wool furnishings. The ambiance is snug, informal, and deeply connected to the wilderness, providing a perfect base camp for seasonal alpine adventures."
        },
        {
            "id": "DOC-05",
            "hotel": "Sunrise B&B",
            "category": "Hotel Descriptions",
            "title": "Sunrise B&B - Cozy Heritage Bed & Breakfast",
            "content": "Sunrise B&B is a charming, budget-friendly heritage bed and breakfast housed in a restored 19th-century Victorian townhouse. Offering a cozy, home-away-from-home atmosphere, this property is decorated with antique wooden furniture, vintage wallpaper, and soft lace curtains. It focuses on personalized hospitality, affordable rates, and local charm. The ambiance is peaceful, intimate, and friendly, appealing to solo backpackers, couples, and budget-conscious history buffs."
        },
        {
            "id": "DOC-06",
            "hotel": "Grand Plaza Hotel",
            "category": "Hotel Descriptions",
            "title": "Grand Plaza - Brand History & Core Philosophy",
            "content": "Founded in 1992, the Grand Plaza Hotel brand has always been synonymous with absolute luxury and top-tier hospitality. Our philosophy revolves around 'anticipatory service'—meeting guest needs before they are explicitly expressed. The hotel boasts curated fine art collections in every corridor, bespoke custom fragrances, and classical piano music in the lounge, ensuring a prestigious experience that matches its high-end brand reputation."
        },
        {
            "id": "DOC-07",
            "hotel": "Seaside Haven Resort",
            "category": "Hotel Descriptions",
            "title": "Seaside Haven - Wellness & Star-Rated Experience",
            "content": "As an award-winning boutique resort, Seaside Haven blends high-end comfort with environmental stewardship. The property utilizes sustainable energy and serves organic, locally sourced seafood. The resort is certified 4-star and features a private beach strip, an infinity edge pool looking out onto the ocean, and a dedicated team of activity coordinators to ensure a memorable vacation."
        },
        {
            "id": "DOC-08",
            "hotel": "Alpine Lodge",
            "category": "Hotel Descriptions",
            "title": "Alpine Lodge - Summer & Winter Seasonal Dynamics",
            "content": "Alpine Lodge operates dynamically across the seasons. In winter, it transforms into a cozy ski-in/ski-out lodge with ski racks and a mudroom. In summer, it pivots to a hiking and mountain biking retreat, offering guided nature trails and equipment storage. This flexible positioning makes it a year-round destination for nature-oriented holidaymakers."
        },
        {
            "id": "DOC-09",
            "hotel": "Hotel X",
            "category": "Hotel Descriptions",
            "title": "Hotel X - Design Concept & Target Demographic",
            "content": "Hotel X targets the modern generation of travelers who value efficiency over traditional luxury. Designed by award-winning architects, the layout maximizes natural light and utilizes ergonomic furniture. The public lounges double as open networking hubs, attracting local entrepreneurs, freelancers, and global business travelers alike."
        },

        # === CATEGORY 2: Amenities (8 docs) ===
        {
            "id": "DOC-10",
            "hotel": "Grand Plaza Hotel",
            "category": "Amenities",
            "title": "Grand Plaza - Luxury Spa, Pool & Dining Services",
            "content": "The Grand Plaza Hotel offers world-class wellness facilities including 'The Obsidian Spa,' which features custom massage therapies, a Finnish sauna, and a steam bath. Guests can swim in the heated indoor lap pool under a glass dome or work out in the fully equipped 24-hour fitness center. Additionally, the hotel offers multiple high-end dining options: a Michelin-starred French restaurant, a sleek cocktail bar, and full-service room dining. All rooms are equipped with free high-speed WiFi, smart flat-screen TVs, automated climate control, marble bathrooms, premium bathrobes, and a complimentary hot breakfast buffet served daily in the main hall."
        },
        {
            "id": "DOC-11",
            "hotel": "Seaside Haven Resort",
            "category": "Amenities",
            "title": "Seaside Haven - Coastal Activities & Beach Amenities",
            "content": "Guests at Seaside Haven Resort enjoy exclusive access to a private sandy beach equipped with complimentary lounge chairs, beach umbrellas, and private cabanas. The resort offers free water sports gear, including paddleboards, kayaks, and snorkeling equipment. The wellness pavilion includes an outdoor ocean-view yoga deck and the 'Sea Breeze Spa' specializing in thalassotherapy. The resort features two restaurants: a beachfront seafood grill and an open-air juice bar. Rooms feature private ocean-facing balconies, hammock swings, ceiling fans, free WiFi, and organic toiletries."
        },
        {
            "id": "DOC-12",
            "hotel": "Hotel X",
            "category": "Amenities",
            "title": "Hotel X - Coworking, Smart Tech & Connectivity",
            "content": "Hotel X provides a full-scale executive business center, shared hot-desking zones, and private video-conferencing pods. The building is equipped with ultra-fast enterprise-grade fiber WiFi throughout the property. Rooms feature smart-room automations controlled via a dedicated mobile app, including keyless entry, lighting presets, and smart temperature adjustments. Other amenities include a 24/7 grab-and-go pantry, charging stations for electric vehicles, and an express laundry service. WiFi is complimentary, but breakfast is not included in the standard room rate."
        },
        {
            "id": "DOC-13",
            "hotel": "Alpine Lodge",
            "category": "Amenities",
            "title": "Alpine Lodge - Hearth & Outdoor Comforts",
            "content": "Amenities at Alpine Lodge center on rustic comfort and outdoor recovery. The centerpiece of the lodge is the great lounge, featuring a massive stone fireplace where hot cocoa is served nightly. Outdoors, guests can soak in a heated cedar hot tub overlooking the forest. The lodge offers ski and snowboard storage lockers, boot dryers, and mountain bike rentals. Rooms are equipped with natural wool blankets, wooden rocking chairs, tea/coffee making stations, and satellite television. WiFi is available in the main lodge, but is intermittent in the outer cabins."
        },
        {
            "id": "DOC-14",
            "hotel": "Sunrise B&B",
            "category": "Amenities",
            "title": "Sunrise B&B - Homemade Dining & Cozy Conveniences",
            "content": "Sunrise B&B prides itself on its hospitality, offering a complimentary home-cooked breakfast every morning, featuring signature buttermilk pancakes, fresh fruit, and organic local coffee. The B&B features a beautiful English cottage garden with outdoor seating where guests can read or drink tea. Inside, the guest parlor offers board games, a book exchange shelf, and free WiFi throughout the house. Rooms are individually heated and include comfortable orthopedic mattresses and soft cotton linens, though bathrooms are shared on each floor."
        },
        {
            "id": "DOC-15",
            "hotel": "Sunrise B&B",
            "category": "Amenities",
            "title": "Sunrise B&B - Complimentary High-Speed Internet & Breakfast",
            "content": "At Sunrise B&B, we believe in keeping travelers connected and well-fed. Every reservation comes with free high-speed wireless internet (WiFi) accessible throughout all guest rooms and common areas. In addition, guests receive our highly-rated complimentary hot breakfast cooked to order every morning, which includes fresh eggs, local jam, pancakes, and selection of herbal teas."
        },
        {
            "id": "DOC-16",
            "hotel": "Grand Plaza Hotel",
            "category": "Amenities",
            "title": "Grand Plaza - Premium Room Features",
            "content": "The rooms at Grand Plaza Hotel are designed to pamper. Each suite includes a king-size pillow-top bed, a dedicated writing desk with ergonomic seating, a stocked minibar, a Nespresso machine, and a multi-jet rainfall shower. Free WiFi is standard across all rooms, and a complimentary breakfast buffet is included for all bookings."
        },
        {
            "id": "DOC-17",
            "hotel": "Hotel X",
            "category": "Amenities",
            "title": "Hotel X - Meeting Rooms & Fitness Facilities",
            "content": "Hotel X has three high-tech boardroom style meeting rooms equipped with interactive whiteboards and 4K displays. The hotel also features a modern fitness studio focusing on high-intensity interval training (HIIT) equipment and stationary bikes. While the hotel offers free WiFi, breakfast is paid."
        },

        # === CATEGORY 3: Guest Reviews (11 docs) ===
        {
            "id": "DOC-18",
            "hotel": "Grand Plaza Hotel",
            "category": "Guest Reviews",
            "title": "Grand Plaza - 5-Star Experience Review",
            "content": "Review Rating: 5/5. 'An absolutely outstanding experience at the Grand Plaza! The service was impeccable from the moment the doorman greeted us. The room was immaculate and quiet, and the marble bathroom felt like a personal spa. The complimentary breakfast buffet had an incredible selection of hot dishes, pastries, and fresh juices. Free WiFi was fast enough for my video conference calls. Highly recommend for luxury travelers!'"
        },
        {
            "id": "DOC-19",
            "hotel": "Grand Plaza Hotel",
            "category": "Guest Reviews",
            "title": "Grand Plaza - Neutral Business Stay Review",
            "content": "Review Rating: 3/5. 'The Grand Plaza is clean and professionally run, but it felt a bit cold and business-like. The dinner at the Michelin restaurant was overpriced, and the pool area was crowded. WiFi and breakfast were free, which was nice, but for the steep price, I expected a bit more character. Good, but not amazing.'"
        },
        {
            "id": "DOC-20",
            "hotel": "Seaside Haven Resort",
            "category": "Guest Reviews",
            "title": "Seaside Haven - Beautiful Beach Escape Review",
            "content": "Review Rating: 5/5. 'This place is heaven! The resort is beautiful, and our room opened right onto the sand. The ocean view from the balcony was stunning. The staff was incredibly warm and helpful, organizing a bonfire for us on the beach. Excellent reviews are totally justified—this is the perfect beach getaway. I will definitely be returning next year!'"
        },
        {
            "id": "DOC-21",
            "hotel": "Seaside Haven Resort",
            "category": "Guest Reviews",
            "title": "Seaside Haven - Excellent Reviews Near the Beach",
            "content": "Review Rating: 5/5. 'I cannot recommend Seaside Haven Resort enough! It has excellent reviews, and it is located literally right on the beach, just steps from the ocean waves. The room had a gorgeous private balcony with a hammock, and the beach-side service was exceptional. The thalassotherapy spa was out of this world. Best beach vacation ever!'"
        },
        {
            "id": "DOC-22",
            "hotel": "Seaside Haven Resort",
            "category": "Guest Reviews",
            "title": "Seaside Haven - Disappointing Pool Service Review",
            "content": "Review Rating: 3/5. 'The location next to the beach is absolutely gorgeous, and the rooms are nicely decorated. However, the poolside service was extremely slow, and the main pool was overcrowded with children. The room WiFi was also quite spotty. Go for the beach, but don't expect fast service.'"
        },
        {
            "id": "DOC-23",
            "hotel": "Hotel X",
            "category": "Guest Reviews",
            "title": "Hotel X - Efficient and Productive Review",
            "content": "Review Rating: 4/5. 'Hotel X was perfect for my business trip. The check-in was automated and took 30 seconds via the kiosk. The coworking area in the lobby is excellent, with high-speed internet and comfortable chairs. The room is small and minimalist but very clean and quiet. Excellent price-to-quality ratio for professionals.'"
        },
        {
            "id": "DOC-24",
            "hotel": "Hotel X",
            "category": "Guest Reviews",
            "title": "Hotel X - Clean But Noisy Review",
            "content": "Review Rating: 2/5. 'The hotel itself is modern and very clean, but my room faced a noisy street, and the thin windows let in all the traffic sounds. The smart app for keyless entry crashed twice, and I had to go to the front desk. Also, note that breakfast is not free here, and it is quite basic for the price. Not a great stay.'"
        },
        {
            "id": "DOC-25",
            "hotel": "Alpine Lodge",
            "category": "Guest Reviews",
            "title": "Alpine Lodge - Cozy Fireplace Review",
            "content": "Review Rating: 5/5. 'We had a lovely winter weekend at Alpine Lodge! Sitting by the massive stone fireplace in the evening with hot chocolate was magical. The room was cozy with warm wood paneling, and the outdoor hot tub under the stars was incredible. Perfect place to relax after skiing. We loved it!'"
        },
        {
            "id": "DOC-26",
            "hotel": "Alpine Lodge",
            "category": "Guest Reviews",
            "title": "Alpine Lodge - Poor Connectivity Review",
            "content": "Review Rating: 2/5. 'The mountains are beautiful, but the lodge was a bit too rustic. There was no cell signal in our cabin, and the WiFi in the main lodge was slow and kept disconnecting. The floors were very creaky, and the heating took a long time to warm up. Only stay here if you want to be completely disconnected.'"
        },
        {
            "id": "DOC-27",
            "hotel": "Sunrise B&B",
            "category": "Guest Reviews",
            "title": "Sunrise B&B - Wonderful Homemade Breakfast Review",
            "content": "Review Rating: 5/5. 'What a lovely place! The host, Sarah, made us feel so welcome. The B&B is close to the train station, and our room was clean and charming. The absolute highlight was the homemade pancake breakfast in the morning—best pancakes I have ever had! The garden is beautiful, and WiFi is free. Shared bathrooms were clean. Excellent value!'"
        },
        {
            "id": "DOC-28",
            "hotel": "Sunrise B&B",
            "category": "Guest Reviews",
            "title": "Sunrise B&B - Loud Train Noise Review",
            "content": "Review Rating: 3/5. 'The breakfast was delicious and the bed was comfortable. However, the B&B is located right next to the active train tracks, and trains run all night. It was very loud, and I had trouble sleeping. If you are a light sleeper, bring earplugs. Otherwise, the hospitality and price are very good.'"
        },

        # === CATEGORY 4: Policies (8 docs) ===
        {
            "id": "DOC-29",
            "hotel": "Grand Plaza Hotel",
            "category": "Policies",
            "title": "Grand Plaza - Check-in, Check-out & General Hours",
            "content": "Standard check-in time at the Grand Plaza Hotel begins at 2:00 PM. Guests arriving earlier may store their luggage at the concierge desk, and early check-in is subject to room availability and a surcharge. Check-out time is strictly at 11:00 AM. Late check-out requests must be approved by the front desk in advance and may incur a fee equivalent to half a day's room rate. Quiet hours are enforced between 10:00 PM and 7:00 AM."
        },
        {
            "id": "DOC-30",
            "hotel": "Grand Plaza Hotel",
            "category": "Policies",
            "title": "Grand Plaza - Cancellation & Prepayment Rules",
            "content": "Reservations at the Grand Plaza Hotel must be guaranteed with a valid credit card. Cancellations must be made at least 72 hours prior to the arrival date to avoid a penalty. Cancellations made within 72 hours of arrival, or failure to check in (no-show), will result in a penalty charge equivalent to the first night's room rate plus taxes."
        },
        {
            "id": "DOC-31",
            "hotel": "Seaside Haven Resort",
            "category": "Policies",
            "title": "Seaside Haven - Pet & Animal Policy",
            "content": "Seaside Haven Resort is a pet-friendly property that welcomes well-behaved dogs. Guests traveling with pets must notify the resort at the time of booking. A maximum of two dogs per room is permitted, with a weight limit of 25 pounds per dog. A non-refundable pet cleaning fee of $75 per stay will be applied to the bill. Pets are not permitted in the dining areas or pool deck, and must be kept on a leash in all common outdoor areas."
        },
        {
            "id": "DOC-32",
            "hotel": "Seaside Haven Resort",
            "category": "Policies",
            "title": "Seaside Haven - Refund & Deposit Policy",
            "content": "A deposit equivalent to one night's room charge is required at the time of booking for Seaside Haven Resort. Full refunds are processed if the reservation is canceled at least 7 days before the arrival date. For cancellations made within the 7-day window, the deposit is forfeited. Refunds will be issued back to the original form of payment within 5 to 7 business days."
        },
        {
            "id": "DOC-33",
            "hotel": "Hotel X",
            "category": "Policies",
            "title": "Hotel X - Cancellation & Late Arrival Policy",
            "content": "The cancellation policy of Hotel X allows guests to cancel their booking free of charge up to 48 hours prior to their scheduled check-in time (which is 3:00 PM). If a booking is canceled within the 48-hour window before arrival, or in case of a no-show, the hotel will charge a penalty fee equivalent to the full cost of the first night's room rate plus applicable local taxes. Non-refundable promotional bookings are excluded from this policy and cannot be refunded or modified."
        },
        {
            "id": "DOC-34",
            "hotel": "Hotel X",
            "category": "Policies",
            "title": "Hotel X - Security & ID Verification Norms",
            "content": "At Hotel X, security and safety are paramount. All guests, including additional occupants, must present a valid government-issued photo ID (passport, driver's license, or national ID card) at check-in. A physical credit card matching the guest name must be presented for a temporary incidental hold of $100. Digital copies or photo-scans of IDs are strictly not accepted."
        },
        {
            "id": "DOC-35",
            "hotel": "Alpine Lodge",
            "category": "Policies",
            "title": "Alpine Lodge - Fireplace Safety & Check-out Steps",
            "content": "To maintain safety in our wooden structure, guests at Alpine Lodge must follow strict fireplace guidelines. Firewood must only be burned inside the designated hearths, and the protective metal screen must remain closed during use. Before checking out, guests must ensure the fire is completely extinguished. Checkout is at 10:00 AM, and guests must leave keys in the drop box at the front cabin."
        },
        {
            "id": "DOC-36",
            "hotel": "Sunrise B&B",
            "category": "Policies",
            "title": "Sunrise B&B - Smoking & Quiet Hours Policies",
            "content": "Sunrise B&B is a strictly 100% smoke-free property. Smoking of any substance, including e-cigarettes and vapes, is prohibited inside all bedrooms, corridors, and common spaces. A cleaning fee of $250 will be charged for violations. Quiet hours are observed from 9:30 PM to 8:00 AM to ensure all guests get a peaceful night's rest."
        },

        # === CATEGORY 5: Location Details (8 docs) ===
        {
            "id": "DOC-37",
            "hotel": "Grand Plaza Hotel",
            "category": "Location Details",
            "title": "Grand Plaza - Proximity to Downtown & Transit",
            "content": "The Grand Plaza Hotel is situated in the downtown financial core, just a 5-minute walk from the Central Subway Station. The historic City Museum, the Opera House, and the Luxury Shopping Promenade are all within a 1-mile radius. Taxis and ride-sharing services are readily available outside the lobby. The airport express train connects the Central Subway Station directly to the International Airport in 25 minutes."
        },
        {
            "id": "DOC-38",
            "hotel": "Grand Plaza Hotel",
            "category": "Location Details",
            "title": "Grand Plaza - Surrounding District & Dining Hubs",
            "content": "Located adjacent to the financial hub, the Grand Plaza is surrounded by upscale dining establishments, rooftop cocktail bars, and international corporate headquarters. It is located exactly 0.5 miles from the Municipal Park, which offers walking trails, lake boating, and botanical glasshouses, providing a green escape within the city."
        },
        {
            "id": "DOC-39",
            "hotel": "Seaside Haven Resort",
            "category": "Location Details",
            "title": "Seaside Haven - Beach Location & Waterfront Access",
            "content": "Seaside Haven Resort is located right on the beach, situated directly on the shores of Silver Beach, just 20 meters from the ocean tide line. The property overlooks a quiet cove, providing calm waters for swimming. The famous Seaside Boardwalk, lined with seafood restaurants, artisan shops, and bicycle rental stations, begins right at the edge of the resort property, offering direct pedestrian access."
        },
        {
            "id": "DOC-40",
            "hotel": "Seaside Haven Resort",
            "category": "Location Details",
            "title": "Seaside Haven - Nearby Attractions & Neighborhood",
            "content": "Located in a scenic coastal neighborhood, Seaside Haven Resort is 1.5 miles from the historic Point Lookout Lighthouse, a popular spot for sunsets and whale watching. The local Seafood Market, where local fishermen bring in the daily catch, is a 10-minute walk down the beach. The resort is 15 miles from the regional airport, and a shuttle service is available for guests."
        },
        {
            "id": "DOC-41",
            "hotel": "Hotel X",
            "category": "Location Details",
            "title": "Hotel X - Proximity to Convention Center & CBD",
            "content": "Hotel X is strategically located in the tech corridor of the Central Business District (CBD). It is positioned directly across the street from the City Convention Center (only 100 meters away), making it extremely convenient for conference attendees. It is surrounded by modern office blocks, quick-service cafes, and shared workspace buildings."
        },
        {
            "id": "DOC-42",
            "hotel": "Alpine Lodge",
            "category": "Location Details",
            "title": "Alpine Lodge - Ski Lifts & National Park Trails",
            "content": "Alpine Lodge is nestled in the mountain pine forests, located just 200 yards from the Peak Express Ski Lift, offering quick walking access to the slopes. The lodge is adjacent to the boundary of the Pine Valley National Park, and several popular hiking trails—ranging from easy forest walks to strenuous peak climbs—begin directly behind the lodge's main cabin."
        },
        {
            "id": "DOC-43",
            "hotel": "Sunrise B&B",
            "category": "Location Details",
            "title": "Sunrise B&B - Historic Town Center & Sightseeing",
            "content": "Sunrise B&B is located in the charming historic district of the town, surrounded by cobbled streets and historical buildings. The Town Museum and the local Craft Market are a 10-minute walk away. Guests can find several traditional pubs and cozy family-run restaurants within three blocks of the house."
        },
        {
            "id": "DOC-44",
            "hotel": "Sunrise B&B",
            "category": "Location Details",
            "title": "Sunrise B&B - Rail Links & Local Transport",
            "content": "Located just 150 meters from the Central Rail Terminal, Sunrise B&B offers exceptionally easy transport links for travelers arriving by train. Local buses stop right at the corner of the street, connecting guests to nearby scenic viewpoints and neighboring villages. The proximity to the station makes it a highly convenient hub for regional exploration."
        }
    ]

    # Save to a single JSON file in the project dataset directory
    dataset_dir = os.path.join(BASE_DIR, "dataset")
    os.makedirs(dataset_dir, exist_ok=True)
    json_path = os.path.join(BASE_DIR, "hotel_dataset.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(documents, f, indent=4, ensure_ascii=False)
    print(f"Saved {len(documents)} documents to {json_path}")

    # Save to individual TXT files in dataset/ directory
    for doc in documents:
        # Create a safe filename
        safe_title = doc["title"].replace(" - ", "_").replace(" & ", "_and_").replace(" ", "_").replace(",", "").replace("'", "")
        filename = os.path.join(dataset_dir, f"{doc['id']}_{safe_title}.txt")
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"ID: {doc['id']}\n")
            f.write(f"Hotel: {doc['hotel']}\n")
            f.write(f"Category: {doc['category']}\n")
            f.write(f"Title: {doc['title']}\n")
            f.write(f"Content: {doc['content']}\n")
    print(f"Saved individual TXT files in {dataset_dir}")

if __name__ == "__main__":
    generate_hotel_dataset()
