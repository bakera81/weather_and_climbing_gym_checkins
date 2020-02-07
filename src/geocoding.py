import googlemaps 
import os
import re

class GMaps():
    
    def __init__(self, gym):
        
        self.gmaps = googlemaps.Client(key=os.environ['GOOGLE_API']) 

        if gym[0].lower() == 'g':
            self.gym_address = '700 Golden Ridge Rd, Golden, CO 80401, USA'
        if gym[0].lower() == 'e':
            self.gym_address = '1050 W Hampden Ave Ste 100, Englewood, CO 80110, USA'
            
    def get_dist(self, address):
        
        result = self.gmaps.distance_matrix(address, self.gym_address) 

        dist_km = result['rows'][0]['elements'][0]['distance']['text']
        duration = result['rows'][0]['elements'][0]['duration']['text']
        
        return self.parse_distance(dist_km), self.parse_duration(duration)
    
    def parse_duration(self, string):
        
        num_list = re.findall(r'\d+', string)[::-1]
        
        duration = 0
        conv = [1, 60, 1440][:len(num_list)]
        for num, conv in zip(num_list, conv):
            duration += float(num) * conv
        
        return duration

    def parse_distance(self, string):
        
        return float(string.split()[0].replace(',', ''))
    
if __name__ == '__main__':
    
    gol_maps = GMaps('gol')
    
    gol_maps.get_dist('2329 S Eldridge St Lakewood CO')    
    gol_maps.get_dist('9755 Horseback Ridge Missoula MT')