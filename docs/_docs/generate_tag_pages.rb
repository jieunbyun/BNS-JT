#!/usr/bin/env ruby

tags = {}

# Loop through all posts
Dir.glob('_posts/*.md') do |post|
    File.readlines(post).each do |line|
        if line =~ /^tags:/
            # Extract tags from post front matter
            tags_in_post = line.sub('tags:', '').strip.split(',').map(&:strip)
            
            # Create tag pages
            tags_in_post.each do |tag|
                # Remove square brackets from tag (if present)
                tag_without_brackets = tag.tr('[]', '')  # Remove square brackets
                
                File.open("tag/#{tag_without_brackets}.html", "w") do |file|
                    file.puts "---"
                    file.puts "layout: tag"  # Use the tag layout
                    file.puts "tag: #{tag_without_brackets}"  # Pass the tag as a variable to the layout
                    file.puts "---"
                end
            end
            
            break
        end
        
    end
end
