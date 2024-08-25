Jekyll::Hooks.register :site, :after_init do |site|
    puts "Hello from the test plugin!"
  end